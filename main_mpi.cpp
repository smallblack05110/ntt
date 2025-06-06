#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <pthread.h>
#include <unistd.h>
#include <mpi.h>

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

// 巴雷特模乘结构体
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
std::string uint128_to_string(__uint128_t value) {
    if (value == 0) return "0";
    char buffer[40];
    int index = 0;
    while (value > 0) {
        buffer[index++] = '0' + static_cast<char>(value % 10);
        value /= 10;
    }
    std::reverse(buffer, buffer + index);
    return std::string(buffer, buffer + index);
}

void fRead(uint64_t *a, uint64_t *b, int *n, int64_t *p, int input_id) {
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

void fWrite(const uint64_t *ab, int n, int input_id) {
    char path_buffer[256];
    sprintf(path_buffer, "files/%d.out", input_id);
    
    ofstream fout;
    fout.open(path_buffer, std::ios::out);
    for (int i = 0; i < n * 2 - 1; ++i) {
        fout << ab[i] << '\n';
    }
    fout.close();
}

void fCheck(uint64_t *ab, int n, int input_id) {
    char path_buffer[256];
    sprintf(path_buffer, "/nttdata/%d.out", input_id);
    
    std::ifstream fin;
    fin.open(path_buffer, std::ios::in);
    for (int i = 0; i < n * 2 - 1; i++) {
        uint64_t x;
        fin >> x;
        if (x != ab[i]) {
            std::cout << "多项式乘法结果错误" << std::endl;
            return;
        }
    }
    std::cout << "多项式乘法结果正确" << std::endl;
    return;
}

__int128_t quick_mod_barrett(__int128_t a, __int128_t b, __int128_t p, const BarrettReduction &barrett) {
    __int128_t res = 1; a %= p;
    while (b > 0) {
        if (b & 1) res = barrett.reduce(res * a);
        a = barrett.reduce(a * a); b >>= 1;
    }
    return res;
}

__int128_t quick_mod(__int128_t a, __int128_t b, __int128_t p) {
    __int128_t res = 1; a %= p;
    while (b > 0) {
        if (b & 1) res = (res * a) % p;
        a = (a * a) % p; b >>= 1;
    }
    return res;
}

// 标准NTT实现（每个进程内部使用）
void ntt_iter_barrett(vector<uint32_t> &a, uint64_t p, int root, bool invert, BarrettReduction &barrett) {
    int n = a.size();
    
    // 位反转置换
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) swap(a[i], a[j]);
    }
    
    // 蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t wn = quick_mod_barrett(root, (p - 1) / len, p, barrett);
        if (invert) wn = quick_mod_barrett(wn, p - 2, p, barrett);
        
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

// MPI分布式多项式乘法
void mpi_polynomial_multiply(vector<uint32_t> &a, vector<uint32_t> &b, vector<uint32_t> &result,
                            uint64_t p, int root, BarrettReduction &barrett, int rank, int size) {
    int n = a.size();
    
    if (rank == 0) {
        cout << "  进程 " << rank << ": 开始NTT变换，数据长度 = " << n << endl;
    }
    
    // 前向NTT
    ntt_iter_barrett(a, p, root, false, barrett);
    ntt_iter_barrett(b, p, root, false, barrett);
    
    // 计算每个进程负责的点乘范围
    int chunk_size = n / size;
    int remainder = n % size;
    int start = rank * chunk_size + min(rank, remainder);
    int end = start + chunk_size + (rank < remainder ? 1 : 0);
    
    if (rank == 0) {
        cout << "  进程 " << rank << ": 负责点乘范围 [" << start << ", " << end << ")" << endl;
    }
    
    // 点乘
    result.resize(n);
    for (int i = start; i < end; ++i) {
        result[i] = barrett.mul_mod(a[i], b[i]);
    }
    
    // 收集所有进程的点乘结果
    vector<int> recvcounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = chunk_size + (i < remainder ? 1 : 0);
        displs[i] = i * chunk_size + min(i, remainder);
    }
    
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                   result.data(), recvcounts.data(), displs.data(), MPI_UINT32_T, MPI_COMM_WORLD);
    
    if (rank == 0) {
        cout << "  进程 " << rank << ": 开始逆NTT变换" << endl;
    }
    
    // 逆向NTT
    ntt_iter_barrett(result, p, root, true, barrett);
    
    // 乘以n的逆元
    uint64_t inv_n = quick_mod_barrett(n, p - 2, p, barrett);
    for (int i = start; i < end; ++i) {
        result[i] = barrett.mul_mod(result[i], inv_n);
    }
    
    // 收集最终结果
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                   result.data(), recvcounts.data(), displs.data(), MPI_UINT32_T, MPI_COMM_WORLD);
}

// MPI分布式CRT重建
void mpi_crt_reconstruct(vector<vector<uint32_t>> &mods, vector<uint64_t> &ab,
                        __uint128_t M, __uint128_t *K, __uint128_t *invK,
                        int64_t p_, int CRT_CNT, vector<BarrettReduction*> &barrett_mods,
                        int rank, int size) {
    int n = mods[0].size();
    int chunk_size = n / size;
    int remainder = n % size;
    int start = rank * chunk_size + min(rank, remainder);
    int end = start + chunk_size + (rank < remainder ? 1 : 0);
    
    if (rank == 0) {
        cout << "  进程 " << rank << ": 开始CRT重建，负责范围 [" << start << ", " << end << ")" << endl;
    }
    
    ab.resize(n);
    
    // 每个进程处理自己的数据块
    for (int i = start; i < end; ++i) {
        __uint128_t sum = 0;
        for (int j = 0; j < CRT_CNT; ++j) {
            __uint128_t term = mods[j][i];
            term = barrett_mods[j]->mul_mod(term, invK[j]);
            term = (term * K[j]) % M;
            sum = (sum + term) % M;
        }
        ab[i] = uint64_t(sum % p_);
    }
    
    // 收集所有进程的CRT结果
    vector<int> recvcounts(size), displs(size);
    for (int i = 0; i < size; ++i) {
        recvcounts[i] = chunk_size + (i < remainder ? 1 : 0);
        displs[i] = i * chunk_size + min(i, remainder);
    }
    
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                   ab.data(), recvcounts.data(), displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);
}

__uint128_t power_barrett(__uint128_t base, __uint32_t exp, __uint32_t mod, BarrettReduction &barrett) {
    __uint128_t res = 1; base %= mod;
    while (exp > 0) {
        if (exp & 1) res = barrett.reduce(res * base);
        base = barrett.reduce(base * base); exp >>= 1;
    }
    return res;
}

__uint128_t modinv_crt_barrett(__uint128_t a, __uint128_t m, BarrettReduction &barrett) {
    return power_barrett(a, m - 2, m, barrett);
}

uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[]) {
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // 只有主进程输出MPI信息
    if (rank == 0) {
        cout << "========== MPI分布式NTT多项式乘法 ==========" << endl;
        cout << "MPI进程数: " << size << endl;
        cout << "OpenMP线程数: " << omp_get_max_threads() << endl;
        cout << "===========================================" << endl;
    }
    
    int test_begin = 0, test_end = 4;
    const int root = 3;
    const int CRT_CNT = 4;
    
    // 根为3的小模数列表
    uint64_t small_mods[CRT_CNT] = {
        469762049ULL, 998244353ULL,
        1004535809ULL, 1224736769ULL
    };
    
    // 为每个小模数创建巴雷特模乘计算器
    vector<BarrettReduction*> barrett_mods;
    for (int i = 0; i < CRT_CNT; ++i) {
        barrett_mods.push_back(new BarrettReduction(small_mods[i]));
    }
    
    // 计算所有小模数乘积 M
    __uint128_t M = 1;
    for (int i = 0; i < CRT_CNT; ++i) M *= small_mods[i];
    
    // 预计算 CRT 常量 K 和 invK
    __uint128_t K[CRT_CNT], invK[CRT_CNT];
    for (int i = 0; i < CRT_CNT; ++i) {
        K[i] = M / small_mods[i];
        invK[i] = modinv_crt_barrett(K[i], small_mods[i], *barrett_mods[i]);
    }
    
    for (int id = test_begin; id <= test_end; ++id) {
        long double ans = 0;
        int n_;
        int64_t p_;
        
        // 只有主进程读取数据
        if (rank == 0) {
            fRead(a, b, &n_, &p_, id);
            cout << "\n测试用例 " << id << ": n=" << n_ << ", p=" << p_ << endl;
        }
        
        // 广播数据大小和模数
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_INT64_T, 0, MPI_COMM_WORLD);
        
        int len = 1;
        while (len < 2 * n_) len <<= 1;
        
        // 广播输入数据
        MPI_Bcast(a, len, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, len, MPI_UINT64_T, 0, MPI_COMM_WORLD);
        
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = chr::high_resolution_clock::now();
        
        // 存储每个小模NTT结果
        vector<vector<uint32_t>> mods(CRT_CNT);
        for (int i = 0; i < CRT_CNT; i++) {
            mods[i].resize(len);
        }
        
        if (rank == 0) {
            cout << "开始并行NTT计算..." << endl;
        }
        
        // 并行处理每个小模数的NTT（在进程内部使用OpenMP）
        #pragma omp parallel for num_threads(min(CRT_CNT, omp_get_max_threads()))
        for (int mod_idx = 0; mod_idx < CRT_CNT; ++mod_idx) {
            // 转换输入数据到当前模数
            vector<uint32_t> a_mod(len), b_mod(len);
            for (int i = 0; i < len; i++) {
                a_mod[i] = static_cast<uint32_t>(barrett_mods[mod_idx]->reduce(a[i]));
                b_mod[i] = static_cast<uint32_t>(barrett_mods[mod_idx]->reduce(b[i]));
            }
            
            // MPI分布式多项式乘法
            if (rank == 0) {
                cout << "处理模数 " << small_mods[mod_idx] << " (索引 " << mod_idx << ")" << endl;
            }
            
            mpi_polynomial_multiply(a_mod, b_mod, mods[mod_idx], 
                                  small_mods[mod_idx], root, *barrett_mods[mod_idx], rank, size);
        }
        
        if (rank == 0) {
            cout << "开始CRT重建..." << endl;
        }
        
        // MPI分布式CRT重建
        vector<uint64_t> ab_vec;
        mpi_crt_reconstruct(mods, ab_vec, M, K, invK, p_, CRT_CNT, barrett_mods, rank, size);
        
        // 转换回数组格式
        for (int i = 0; i < 2 * n_ - 1; ++i) {
            ab[i] = ab_vec[i];
        }
        
        // 最终模数约简
        BarrettReduction barrett_final(p_);
        int chunk_size = (2 * n_ - 1) / size;
        int remainder = (2 * n_ - 1) % size;
        int start_idx = rank * chunk_size + min(rank, remainder);
        int end_idx = start_idx + chunk_size + (rank < remainder ? 1 : 0);
        
        for (int i = start_idx; i < end_idx; ++i) {
            ab[i] = barrett_final.reduce(ab[i]);
        }
        
        // 收集最终结果
        vector<int> recvcounts(size), displs(size);
        for (int i = 0; i < size; ++i) {
            recvcounts[i] = chunk_size + (i < remainder ? 1 : 0);
            displs[i] = i * chunk_size + min(i, remainder);
        }
        
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, 
                       ab, recvcounts.data(), displs.data(), MPI_UINT64_T, MPI_COMM_WORLD);
        
        // 同步所有进程
        MPI_Barrier(MPI_COMM_WORLD);
        auto end = chr::high_resolution_clock::now();
        ans = chr::duration<double, ratio<1, 1000>>(end - start).count();
        
        // 只有主进程进行检查和输出
        if (rank == 0) {
            fCheck(ab, n_, id);
            cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " ms" << endl;
            fWrite(ab, n_, id);
            cout << "----------------------------------------" << endl;
        }
    }
    
    // 释放资源
    for (auto &b : barrett_mods) {
        delete b;
    }
    
    if (rank == 0) {
        cout << "所有测试完成！" << endl;
    }
    
    // 结束MPI
    MPI_Finalize();
    return 0;
}