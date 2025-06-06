#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <mpi.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <unistd.h>

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

// 优化的NTT实现 - 改进内存访问模式和减少模乘操作
void ntt_iter_barrett_optimized(vector<uint32_t> &a, uint64_t p, int root, bool invert, BarrettReduction &barrett)
{
    int n = a.size();
    
    // 位反转置换 - 优化版本
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j |= bit;
        if (i < j) swap(a[i], a[j]);
    }
    
    // 预计算单位根的幂 - 减少重复计算
    vector<uint32_t> w_powers(n/2);
    
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t wn = quick_mod_barrett(root, (p - 1) / len, p, barrett);
        if (invert) wn = quick_mod_barrett(wn, p - 2, p, barrett);
        
        // 预计算本轮所需的单位根幂次
        int half_len = len / 2;
        w_powers[0] = 1;
        for (int j = 1; j < half_len; ++j) {
            w_powers[j] = barrett.mul_mod(w_powers[j-1], wn);
        }
        
        // 并行化蝶形运算
        #pragma omp parallel for if(len >= 1024)
        for (int i = 0; i < n; i += len) {
            for (int j = 0; j < half_len; ++j) {
                uint32_t u = a[i + j];
                uint64_t v = barrett.mul_mod(a[i + j + half_len], w_powers[j]);
                
                // 优化加法和减法
                a[i + j] = (u + v >= p) ? u + v - p : u + v;
                a[i + j + half_len] = (u >= v) ? u - v : u + p - v;
            }
        }
    }
}

// 优化的CRT模逆
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

// 优化的MPI数据分发和收集
class MPIDataManager {
public:
    int rank, size;
    vector<int> sendcounts, displs;
    MPIDataManager(int r, int s) : rank(r), size(s) {}
    
    void setup_distribution(int total_size) {
        sendcounts.resize(size);
        displs.resize(size);
        
        int base_size = total_size / size;
        int remainder = total_size % size;
        
        for (int i = 0; i < size; ++i) {
            sendcounts[i] = base_size + (i < remainder ? 1 : 0);
            displs[i] = (i == 0) ? 0 : displs[i-1] + sendcounts[i-1];
        }
    }
    
    void allgatherv_data(void* data, MPI_Datatype datatype) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, data, 
                       sendcounts.data(), displs.data(), datatype, MPI_COMM_WORLD);
    }
    
    int get_local_start() const { return displs[rank]; }
    int get_local_size() const { return sendcounts[rank]; }
};

uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
    // 初始化MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0) {
        cout << "MPI进程数: " << size << endl;
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

    // 创建MPI数据管理器
    MPIDataManager data_manager(rank, size);

    for (int id = test_begin; id <= test_end; ++id) {
        long double ans = 0;
        int n_;
        int64_t p_;
        
        // 只有rank 0读取数据
        if (rank == 0) {
            fRead(a, b, &n_, &p_, id);
        }
        
        // 广播数据大小和模数
        MPI_Bcast(&n_, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&p_, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        
        int len = 1; 
        while (len < 2 * n_) len <<= 1;
        
        if (rank == 0) {
            fill(a + n_, a + len, 0);
            fill(b + n_, b + len, 0);
        }
        
        // 广播输入数据
        MPI_Bcast(a, len, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        MPI_Bcast(b, len, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
        
        auto start = chr::high_resolution_clock::now();

        // 优化1: 减少通信 - 每个进程计算完整的CRT，而不是分发小模数
        // 每个进程计算所有小模数的NTT，但只处理部分数据点
        data_manager.setup_distribution(len);
        int local_start = data_manager.get_local_start();
        int local_size = data_manager.get_local_size();
        
        vector<uint64_t> local_results(local_size);
        
        // 每个进程处理自己的数据段
        for (int i = 0; i < local_size; ++i) {
            int global_idx = local_start + i;
            __uint128_t sum = 0;
            
            // 对每个小模数进行NTT计算
            for (int t = 0; t < CRT_CNT; ++t) {
                vector<uint32_t> a_vec(len), b_vec(len);
                
                // 转换为对应模数下的32位数据
                for (int j = 0; j < len; j++) {
                    a_vec[j] = static_cast<uint32_t>(barrett_mods[t]->reduce(a[j]));
                    b_vec[j] = static_cast<uint32_t>(barrett_mods[t]->reduce(b[j]));
                }
                
                // 执行优化的NTT
                ntt_iter_barrett_optimized(a_vec, small_mods[t], root, false, *barrett_mods[t]);
                ntt_iter_barrett_optimized(b_vec, small_mods[t], root, false, *barrett_mods[t]);
                
                // 点乘
                for (int j = 0; j < len; ++j) {
                    a_vec[j] = barrett_mods[t]->mul_mod(a_vec[j], b_vec[j]);
                }
                
                // 逆NTT
                ntt_iter_barrett_optimized(a_vec, small_mods[t], root, true, *barrett_mods[t]);
                
                // 乘以 n^{-1}
                uint64_t inv_n = quick_mod_barrett(len, small_mods[t] - 2, small_mods[t], *barrett_mods[t]);
                for (int j = 0; j < len; ++j) {
                    a_vec[j] = barrett_mods[t]->mul_mod(a_vec[j], inv_n);
                }
                
                // CRT重建 - 只计算当前进程负责的点
                __uint128_t term = a_vec[global_idx];
                term = barrett_mods[t]->mul_mod(term, invK[t]);
                term = (term * K[t]) % M;
                sum = (sum + term) % M;
            }
            
            local_results[i] = uint64_t(sum % p_);
        }
        
        // 收集所有进程的结果
        fill(ab, ab + len, 0);
        MPI_Allgatherv(local_results.data(), local_size, MPI_UNSIGNED_LONG_LONG,
                       ab, data_manager.sendcounts.data(), data_manager.displs.data(),
                       MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD);
        
        auto end = chr::high_resolution_clock::now();
        ans = chr::duration<double, ratio<1, 1000>>(end - start).count();

        if (rank == 0) {
            fCheck(ab, n_, id);
            cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
            fWrite(ab, n_, id);
        }
    }

    // 释放动态分配的巴雷特模乘计算器
    for (auto &b : barrett_mods) {
        delete b;
    }
    
    MPI_Finalize();
    return 0;
}