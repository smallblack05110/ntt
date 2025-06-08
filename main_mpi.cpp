#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <mpi.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <unistd.h>

// 使用命名空间但避免使用 string
using std::cout;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::vector;
using std::fill;
using std::min;
using std::swap;
using std::reverse;
namespace chr = std::chrono;
using std::ratio;

// 修复的巴雷特模乘结构体
struct BarrettReduction {
    uint64_t mod;
    uint64_t mu;
    uint64_t shift;

    BarrettReduction(uint64_t _mod) : mod(_mod) {
        shift = 64;
        mu = (static_cast<__uint128_t>(1) << shift) / mod;
    }

    inline uint64_t reduce(uint64_t a) const {
        if (a < mod) return a;
        
        uint64_t q = (static_cast<__uint128_t>(a) * mu) >> shift;
        uint64_t r = a - q * mod;
        
        return r < mod ? r : r - mod;
    }

    inline uint64_t mul_mod(uint64_t a, uint64_t b) const {
        __uint128_t prod = static_cast<__uint128_t>(a) * b;
        uint64_t q = (static_cast<__uint128_t>(prod) * mu) >> shift;
        uint64_t r = prod - q * mod;
        
        return r < mod ? r : r - mod;
    }
};

// uint128 转字符串（输出用）
std::string uint128_to_string(__uint128_t value)
{
    if (value == 0) {
        return "0";
    }
    char buffer[40];
    int index = 0;
    while (value > 0) {
        buffer[index++] = '0' + static_cast<char>(value % 10);
        value /= 10;
    }
    std::reverse(buffer, buffer + index);
    return std::string(buffer, buffer + index);
}

void fRead(uint64_t *a, uint64_t *b, int *n, int64_t *p, int input_id)
{
    char path_buffer[256];
    sprintf(path_buffer, "/nttdata/%d.in", input_id);
    ifstream fin;
    fin.open(path_buffer, std::ios::in);
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; ++i) {
        fin >> b[i];
    }
    fin.close();
}

void fWrite(const uint64_t *ab, int n, int input_id)
{
    char path_buffer[256];
    sprintf(path_buffer, "files/%d.out", input_id);
    
    ofstream fout;
    fout.open(path_buffer, std::ios::out);
    for (int i = 0; i < n * 2 - 1; ++i) {
        fout << ab[i] << '\n';
    }
    fout.close();
}

void fCheck(uint64_t *ab, int n, int input_id){
    char path_buffer[256];
    sprintf(path_buffer, "/nttdata/%d.out", input_id);
    
    std::ifstream fin;
    fin.open(path_buffer, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++){
        uint64_t x;
        fin>>x;
        if(x != ab[i]){
            std::cout<<"多项式乘法结果错误"<<std::endl;
            return;
        }
    }
    std::cout<<"多项式乘法结果正确"<<std::endl;
    return;
}

// 使用巴雷特模乘的快速幂
__int128_t quick_mod_barrett(__int128_t a, __int128_t b, __int128_t p, const BarrettReduction &barrett)
{
    __int128_t res = 1; a %= p;
    while (b > 0) {
        if (b & 1) res = barrett.reduce(res * a);
        a = barrett.reduce(a * a); b >>= 1;
    }
    return res;
}

// 优化的NTT实现 - 只在合适的地方使用OpenMP
void ntt_iter_barrett(vector<uint32_t> &a, uint64_t p, int root, bool invert, BarrettReduction &barrett)
{
    int n = a.size();
    
    // 位逆序 - 保持串行，避免复杂的同步
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) swap(a[i], a[j]);
    }
    
    // NTT变换 - 只在大规模时才并行化
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t wn = quick_mod_barrett(root, (p - 1) / len, p, barrett);
        if (invert) wn = quick_mod_barrett(wn, p - 2, p, barrett);
        
        // 只有当工作量足够大时才使用OpenMP
        if (n >= 8192 && len >= 64) {
            #pragma omp parallel for schedule(static) if(n/len >= 4)
            for (int i = 0; i < n; i += len) {
                uint32_t w = 1;
                // 预计算w的幂次，避免重复计算
                for (int k = 0; k < len / 2; ++k) {
                    if (k > 0) w = barrett.mul_mod(w, wn);
                    
                    uint32_t u = a[i + k];
                    uint64_t v = barrett.mul_mod(a[i + k + len / 2], w);
                    
                    uint32_t sum = u + v;
                    if (sum >= p) sum -= p;
                    
                    uint64_t diff = u;
                    if (u < v) diff += p;
                    diff -= v;
                    
                    a[i + k] = sum;
                    a[i + k + len/2] = diff;
                }
            }
        } else {
            // 小规模数据使用串行版本
            for (int i = 0; i < n; i += len) {
                uint32_t w = 1;
                for (int j = 0; j < len / 2; ++j) {
                    uint32_t u = a[i + j];
                    uint64_t v = barrett.mul_mod(a[i + j + len / 2], w);
                    
                    uint32_t sum = u + v;
                    if (sum >= p) sum -= p;
                    
                    uint64_t diff = u;
                    if (u < v) diff += p;
                    diff -= v;
                    
                    a[i + j] = sum;
                    a[i + j + len/2] = diff;
                    
                    w = barrett.mul_mod(w, wn);
                }
            }
        }
    }
}

// 优化的NTT计算函数
void mpi_ntt_compute(vector<uint32_t> &a_local, vector<uint32_t> &b_local, 
                     vector<uint32_t> &result_local, uint64_t p, int root, 
                     BarrettReduction &barrett)
{
    // 对本地数据进行NTT
    ntt_iter_barrett(a_local, p, root, false, barrett);
    ntt_iter_barrett(b_local, p, root, false, barrett);
    
    // 点乘 - 只有数据量大时才并行化
    result_local.resize(a_local.size());
    size_t data_size = a_local.size();
    
    if (data_size >= 4096) {
        #pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < data_size; ++i) {
            result_local[i] = barrett.mul_mod(a_local[i], b_local[i]);
        }
    } else {
        for (size_t i = 0; i < data_size; ++i) {
            result_local[i] = barrett.mul_mod(a_local[i], b_local[i]);
        }
    }
    
    // 逆变换
    ntt_iter_barrett(result_local, p, root, true, barrett);
    
    // 乘以 n^{-1}
    uint64_t inv_n = quick_mod_barrett(data_size, p - 2, p, barrett);
    if (data_size >= 4096) {
        #pragma omp parallel for simd schedule(static)
        for (size_t i = 0; i < data_size; ++i) {
            result_local[i] = barrett.mul_mod(result_local[i], inv_n);
        }
    } else {
        for (size_t i = 0; i < data_size; ++i) {
            result_local[i] = barrett.mul_mod(result_local[i], inv_n);
        }
    }
}

// CRT模逆
__uint128_t power_barrett(__uint128_t base, __uint32_t exp, __uint32_t mod, BarrettReduction &barrett)
{
    __uint128_t res = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) res = barrett.reduce(res * base);
        base = barrett.reduce(base * base); exp >>= 1;
    }
    return res;
}

__uint128_t modinv_crt_barrett(__uint128_t a, __uint128_t m, BarrettReduction &barrett)
{
    return power_barrett(a, m - 2, m, barrett);
}

// 优化的CRT重建函数
void mpi_crt_reconstruction(vector<vector<uint32_t>> &mods, uint64_t *ab, 
                           int start_idx, int end_idx, __uint128_t M, 
                           __uint128_t *K, __uint128_t *invK, int64_t p_, 
                           int CRT_CNT, vector<BarrettReduction*> &barrett_mods)
{
    int work_size = end_idx - start_idx;
    
    // 只有工作量足够大时才并行化
    if (work_size >= 1024) {
        #pragma omp parallel for schedule(static)
        for (int i = start_idx; i < end_idx; ++i) {
            __uint128_t sum = 0;
            for (int j = 0; j < CRT_CNT; ++j) {
                __uint128_t term = mods[j][i];
                term = barrett_mods[j]->mul_mod(term, invK[j]);
                term = (term * K[j]) % M;
                sum = (sum + term) % M;
            }
            ab[i] = uint64_t(sum % p_);
        }
    } else {
        for (int i = start_idx; i < end_idx; ++i) {
            __uint128_t sum = 0;
            for (int j = 0; j < CRT_CNT; ++j) {
                __uint128_t term = mods[j][i];
                term = barrett_mods[j]->mul_mod(term, invK[j]);
                term = (term * K[j]) % M;
                sum = (sum + term) % M;
            }
            ab[i] = uint64_t(sum % p_);
        }
    }
}

uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
    // 简化MPI初始化，避免不必要的线程支持开销
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 智能设置OpenMP线程数
    int num_threads = std::min(4, omp_get_max_threads()); // 限制线程数，避免过度并行
    omp_set_num_threads(num_threads);
    
    if (rank == 0) {
        cout << "MPI进程数: " << size << endl;
        cout << "每个进程OpenMP线程数: " << num_threads << endl;
    }

    int test_begin = 0, test_end = 4;
    const int root = 3;
    const int CRT_CNT = 4;

    uint64_t small_mods[CRT_CNT] = {
        469762049ULL, 998244353ULL,
        1004535809ULL, 1224736769ULL
    };

    vector<BarrettReduction*> barrett_mods;
    for (int i = 0; i < CRT_CNT; ++i) {
        barrett_mods.push_back(new BarrettReduction(small_mods[i]));
    }

    __uint128_t M = 1;
    for (int i = 0; i < CRT_CNT; ++i) M *= small_mods[i];

    __uint128_t K[CRT_CNT], invK[CRT_CNT];
    for (int i = 0; i < CRT_CNT; ++i) {
        K[i] = M / small_mods[i];
        invK[i] = modinv_crt_barrett(K[i], small_mods[i], *barrett_mods[i]);
    }

    for (int id = test_begin; id <= test_end; ++id) {
        long double ans = 0;
        int n_;
        int64_t p_;
        
        if (rank == 0) {
            fRead(a, b, &n_, &p_, id);
        }
        
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        
        int len = 1; 
        while (len < 2 * n_) len <<= 1;
        
        if (rank == 0) {
            fill(a + n_, a + len, 0);
            fill(b + n_, b + len, 0);
        }
        
        MPI_Bcast(a, len, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, len, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        
        auto start = chr::high_resolution_clock::now();

        vector<vector<uint32_t>> mods(CRT_CNT);
        for (int i = 0; i < CRT_CNT; i++) {
            mods[i].resize(len);
        }

        // 简化的工作分配 - 避免复杂的线程同步
        for (int t = 0; t < CRT_CNT; ++t) {
            if (t % size == rank) {
                vector<uint32_t> a_vec(len), b_vec(len);
                
                // 数据转换 - 可以并行化
                #pragma omp parallel for if(len >= 4096)
                for (int i = 0; i < len; i++) {
                    a_vec[i] = static_cast<uint32_t>(barrett_mods[t]->reduce(a[i]));
                    b_vec[i] = static_cast<uint32_t>(barrett_mods[t]->reduce(b[i]));
                }
                
                vector<uint32_t> result_vec;
                mpi_ntt_compute(a_vec, b_vec, result_vec, small_mods[t], root, *barrett_mods[t]);
                
                mods[t] = std::move(result_vec);
            }
        }
        
        // MPI通信保持不变
        for (int t = 0; t < CRT_CNT; ++t) {
            int owner_rank = t % size;
            
            if (rank == owner_rank) {
                for (int dest = 0; dest < size; ++dest) {
                    if (dest != rank) {
                        MPI_Send(mods[t].data(), len, MPI_UNSIGNED, dest, t, MPI_COMM_WORLD);
                    }
                }
            } else {
                MPI_Recv(mods[t].data(), len, MPI_UNSIGNED, owner_rank, t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        
        int data_per_process = len / size;
        int extra_data = len % size;
        int start_idx = rank * data_per_process + min(rank, extra_data);
        int end_idx = start_idx + data_per_process + (rank < extra_data ? 1 : 0);
        
        mpi_crt_reconstruction(mods, ab, start_idx, end_idx, M, K, invK, p_, CRT_CNT, barrett_mods);
        
        vector<int> recvcounts(size), displs(size);
        for (int i = 0; i < size; ++i) {
            int local_data = data_per_process + (i < extra_data ? 1 : 0);
            recvcounts[i] = local_data;
            displs[i] = (i == 0) ? 0 : displs[i-1] + recvcounts[i-1];
        }
        
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, ab, recvcounts.data(), 
                       displs.data(), MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
        
        BarrettReduction barrett_final(p_);
        
        // 最后的还原操作 - 只有数据量大时才并行化
        if (n_ >= 2048) {
            #pragma omp parallel for simd schedule(static)
            for (int i = 0; i < 2 * n_ - 1; ++i) {
                ab[i] = barrett_final.reduce(ab[i]);
            }
        } else {
            for (int i = 0; i < 2 * n_ - 1; ++i) {
                ab[i] = barrett_final.reduce(ab[i]);
            }
        }
        
        auto end = chr::high_resolution_clock::now();
        ans = chr::duration<double, ratio<1, 1000>>(end - start).count();

        if (rank == 0) {
            fCheck(ab, n_, id);
            cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
            fWrite(ab, n_, id);
        }
    }

    for (auto &b : barrett_mods) {
        delete b;
    }
    
    MPI_Finalize();
    return 0;
}
