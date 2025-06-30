#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

class MontMul
{
private:
  uint64_t N;
  uint64_t R;
  int logR;
  uint64_t N_inv_neg;
  uint64_t R2;

  struct EgcdResult
  {
    int64_t g;
    int64_t x;
    int64_t y;
  };

  static EgcdResult egcd(uint64_t a, uint64_t b)
  {
    uint64_t old_r = a, r = b;
    int64_t old_s = 1, s = 0;
    int64_t old_t = 0, t = 1;
    while (r != 0)
    {
      uint64_t quotient = old_r / r;
      uint64_t temp = old_r;
      old_r = r;
      r = temp - quotient * r;

      int64_t temp_s = old_s;
      old_s = s;
      s = temp_s - static_cast<int64_t>(quotient) * s;

      int64_t temp_t = old_t;
      old_t = t;
      t = temp_t - static_cast<int64_t>(quotient) * t;
    }
    return {static_cast<int64_t>(old_r), old_s, old_t};
  }

  static uint64_t modinv(uint64_t a, uint64_t m)
  {
    auto result = egcd(a, m);
    if (result.g != 1)
    {
      throw std::runtime_error("modular inverse does not exist");
    }
    int64_t x = result.x % static_cast<int64_t>(m);
    if (x < 0)
    {
      x += m;
    }
    return static_cast<uint64_t>(x);
  }

public:
  // 构造函数要求 R 为 2 的幂
  MontMul(uint64_t R, uint64_t N) : R(R), N(N)
  {
    if (R == 0 || (R & (R - 1)) != 0)
    {
      throw std::invalid_argument("R must be a power of two");
    }
    logR = static_cast<int>(std::log2(R));
    if ((1ULL << logR) != R)
    {
      throw std::invalid_argument("R is not a power of two");
    }
    uint64_t N_inv = modinv(N, R);
    N_inv_neg = R - N_inv;
    __int128 R_squared = static_cast<__int128>(R) * R;
    R2 = static_cast<uint64_t>(R_squared % N);
  }

  // 获取Montgomery参数，用于GPU计算
  uint64_t getN() const { return N; }
  uint64_t getR() const { return R; }
  int getLogR() const { return logR; }
  uint64_t getNInvNeg() const { return N_inv_neg; }
  uint64_t getR2() const { return R2; }

  // REDC 算法，将 __int128 类型的 T 转换为 Montgomery 域内元素
  uint64_t REDC(__int128 T) const
  {
    uint64_t mask = (logR == 64) ? ~0ULL : ((1ULL << logR) - 1);
    uint64_t m_part = static_cast<uint64_t>(T) & mask;
    uint64_t m = (m_part * N_inv_neg) & mask;
    __int128 mN = static_cast<__int128>(m) * N;
    __int128 t_val = (T + mN) >> logR;
    uint64_t t = static_cast<uint64_t>(t_val);
    return t >= N ? t - N : t;
  }

  // 将普通整数转换到 Montgomery 域
  uint64_t toMont(uint64_t a) const
  {
    return REDC(a * R2);
  }

  // 从 Montgomery 域转换回普通整数
  uint64_t fromMont(uint64_t aR) const
  {
    return REDC(aR);
  }

  // 在 Montgomery 域内进行乘法运算
  uint64_t mulMont(uint64_t aR, uint64_t bR) const
  {
    return REDC(aR * bR);
  }

  // 保持原有接口：对于 a, b（要求均小于模 N），返回 a * b mod N
  uint64_t ModMul(uint64_t a, uint64_t b)
  {
    if (a >= N || b >= N)
    {
      throw std::invalid_argument("input must be smaller than modulus N");
    }
    uint64_t aR = toMont(a);
    uint64_t bR = toMont(b);
    uint64_t abR = mulMont(aR, bR);
    return fromMont(abR);
  }
};

// GPU版本的Montgomery乘法结构体
struct DeviceMontMul {
    uint64_t N;
    uint64_t R;
    int logR;
    uint64_t N_inv_neg;
    uint64_t R2;
};

// CUDA设备函数：REDC算法
__device__ uint64_t device_REDC(unsigned __int128 T, const DeviceMontMul& mont) {
    uint64_t mask = (mont.logR == 64) ? ~0ULL : ((1ULL << mont.logR) - 1);
    uint64_t m_part = static_cast<uint64_t>(T) & mask;
    uint64_t m = (m_part * mont.N_inv_neg) & mask;
    unsigned __int128 mN = static_cast<unsigned __int128>(m) * mont.N;
    unsigned __int128 t_val = (T + mN) >> mont.logR;
    uint64_t t = static_cast<uint64_t>(t_val);
    return t >= mont.N ? t - mont.N : t;
}

// CUDA设备函数：转换到Montgomery域
__device__ uint64_t device_toMont(uint64_t a, const DeviceMontMul& mont) {
    return device_REDC(static_cast<unsigned __int128>(a) * mont.R2, mont);
}

// CUDA设备函数：从Montgomery域转换回来
__device__ uint64_t device_fromMont(uint64_t aR, const DeviceMontMul& mont) {
    return device_REDC(aR, mont);
}

// CUDA设备函数：Montgomery域内乘法
__device__ uint64_t device_mulMont(uint64_t aR, uint64_t bR, const DeviceMontMul& mont) {
    return device_REDC(static_cast<unsigned __int128>(aR) * bR, mont);
}

// Host函数：快速模幂
long long quick_mod(long long a, long long b, long long p) {
    long long result = 1;
    a = a % p;
    while (b > 0) {
        if (b % 2 == 1) {
            result = (result * a) % p;
        }
        a = (a * a) % p;
        b /= 2;
    }
    return result;
}

// CUDA设备函数：快速模幂
__device__ long long device_quick_mod(long long a, long long b, long long p) {
    long long result = 1;
    a = a % p;
    while (b > 0) {
        if (b % 2 == 1) {
            result = (result * a) % p;
        }
        a = (a * a) % p;
        b /= 2;
    }
    return result;
}

// GPU核函数：位反转重排
__global__ void bit_reverse_kernel(long long* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int j = 0;
    int temp_idx = idx;
    int bit = n >> 1;
    
    while (bit > 0) {
        if (temp_idx & 1) {
            j |= bit;
        }
        temp_idx >>= 1;
        bit >>= 1;
    }
    
    if (idx < j) {
        long long temp = a[idx];
        a[idx] = a[j];
        a[j] = temp;
    }
}

// GPU核函数：NTT蝶形运算
__global__ void ntt_butterfly_kernel(long long* a, int n, int len, long long wnR, 
                                    int p, DeviceMontMul mont) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_len = len / 2;
    int num_groups = n / len;
    int total_operations = num_groups * half_len;
    
    if (idx >= total_operations) return;
    
    int group_id = idx / half_len;
    int in_group_id = idx % half_len;
    
    int i = group_id * len + in_group_id;
    int j = i + half_len;
    
    // 计算w^in_group_id，使用快速幂方法
    long long w = device_toMont(1, mont);
    if (in_group_id > 0) {
        long long base = wnR;
        int exp = in_group_id;
        while (exp > 0) {
            if (exp & 1) {
                w = device_mulMont(w, base, mont);
            }
            base = device_mulMont(base, base, mont);
            exp >>= 1;
        }
    }
    
    long long u = a[i];
    long long v = device_mulMont(w, a[j], mont);
    
    a[i] = (u + v) % p;
    a[j] = (u - v + p) % p;
}

// GPU核函数：点乘
__global__ void pointwise_mul_kernel(long long* a, long long* b, long long* c, 
                                   int n, DeviceMontMul mont) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = device_mulMont(a[idx], b[idx], mont);
    }
}

// GPU核函数：最终结果归一化
__global__ void normalize_kernel(long long* c, int n, long long invR, 
                                DeviceMontMul mont) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = device_mulMont(c[idx], invR, mont);
    }
}

// GPU核函数：转换到Montgomery域
__global__ void to_mont_kernel(long long* a, int n, DeviceMontMul mont) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = device_toMont(a[idx], mont);
    }
}

// GPU核函数：从Montgomery域转换回来
__global__ void from_mont_kernel(long long* a, int n, DeviceMontMul mont) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = device_fromMont(a[idx], mont);
    }
}

// GPU版本的NTT
void gpu_ntt_iter(long long* d_a, int n, int p, int root, bool invert, 
                 const DeviceMontMul& mont, const MontMul& mont_cpu) {
    
    // 位反转重排
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    bit_reverse_kernel<<<blocks, threads_per_block>>>(d_a, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 迭代进行蝶形运算
    for (int len = 2; len <= n; len <<= 1) {
        long long wn = quick_mod(root, (p - 1) / len, p);
        if (invert) {
            wn = quick_mod(wn, p - 2, p);
        }
        
        // 在CPU上计算wnR，然后传递给GPU
        long long wnR = mont_cpu.toMont(wn);
        
        int half_len = len / 2;
        int num_groups = n / len;
        int total_operations = num_groups * half_len;
        int blocks_butterfly = (total_operations + threads_per_block - 1) / threads_per_block;
        
        ntt_butterfly_kernel<<<blocks_butterfly, threads_per_block>>>(
            d_a, n, len, wnR, p, mont);
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

vector<long long> gpu_get_result(vector<long long> &a, vector<long long> &b, 
                                int p, int root, const MontMul &mont_cpu) {
    int n = a.size();
    
    // 创建GPU版本的Montgomery乘法参数
    DeviceMontMul mont;
    mont.N = mont_cpu.getN();
    mont.R = mont_cpu.getR();
    mont.logR = mont_cpu.getLogR();
    mont.N_inv_neg = mont_cpu.getNInvNeg();
    mont.R2 = mont_cpu.getR2();
    
    // 分配GPU内存
    long long *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(long long)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(long long)));
    
    // 复制数据到GPU
    CUDA_CHECK(cudaMemcpy(d_a, a.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, b.data(), n * sizeof(long long), cudaMemcpyHostToDevice));
    
    // GPU线程配置
    int threads_per_block = 256;
    int blocks = (n + threads_per_block - 1) / threads_per_block;
    
    // 前向NTT
    gpu_ntt_iter(d_a, n, p, root, false, mont, mont_cpu);
    gpu_ntt_iter(d_b, n, p, root, false, mont, mont_cpu);
    
    // 点乘
    pointwise_mul_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, n, mont);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 反向NTT
    gpu_ntt_iter(d_c, n, p, root, true, mont, mont_cpu);
    
    // 归一化
    long long inv_n = quick_mod(n, p - 2, p);
    long long invR = mont_cpu.toMont(inv_n);
    normalize_kernel<<<blocks, threads_per_block>>>(d_c, n, invR, mont);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // 复制结果回CPU
    vector<long long> c(n);
    CUDA_CHECK(cudaMemcpy(c.data(), d_c, n * sizeof(long long), cudaMemcpyDeviceToHost));
    
    // 释放GPU内存
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    
    return c;
}

void fRead(int *a, int *b, int *n, int *p, int input_id)
{
  string str1 = "./nttdata/";
  string str2 = to_string(input_id);
  string strin = str1 + str2 + ".in";
  char data_path[strin.size() + 1];
  copy(strin.begin(), strin.end(), data_path);
  data_path[strin.size()] = '\0';
  ifstream fin;
  fin.open(data_path, ios::in);
  fin >> *n >> *p;
  for (int i = 0; i < *n; ++i)
  {
    fin >> a[i];
  }
  for (int i = 0; i < *n; ++i)
  {
    fin >> b[i];
  }
  fin.close();
}

void fWrite(int *ab, int n, int input_id)
{
  string str1 = "files/";
  string str2 = to_string(input_id);
  string strout = str1 + str2 + ".out";
  char output_path[strout.size() + 1];
  copy(strout.begin(), strout.end(), output_path);
  output_path[strout.size()] = '\0';
  ofstream fout;
  fout.open(output_path, ios::out);
  for (int i = 0; i < n * 2 - 1; ++i)
  {
    fout << ab[i] << '\n';
  }
  fout.close();
}

void fCheck(int *ab, int n, int input_id)
{
  string str1 = "./nttdata/";
  string str2 = to_string(input_id);
  string strout = str1 + str2 + ".out";
  char data_path[strout.size() + 1];
  copy(strout.begin(), strout.end(), data_path);
  data_path[strout.size()] = '\0';
  ifstream fin;
  fin.open(data_path, ios::in);
  for (int i = 0; i < n * 2 - 1; ++i)
  {
    int x;
    fin >> x;
    if (x != ab[i])
    {
      cout << "多项式乘法结果错误" << endl;
      fin.close();
      return;
    }
  }
  cout << "多项式乘法结果正确" << endl;
  fin.close();
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
    // 初始化CUDA设备
    CUDA_CHECK(cudaSetDevice(0));
    
    int test_begin = 0, test_end = 3;
    for (int id = test_begin; id <= test_end; ++id)
    {
        double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, id);

        int len = 1;
        while (len < 2 * n_)
            len <<= 1;
        fill(a + n_, a + len, 0);
        fill(b + n_, b + len, 0);

        vector<long long> va(a, a + len), vb(b, b + len);
        long long R = 1LL << 30;
        MontMul mont(R, p_);
        
        // 转换到Montgomery域
        for (int i = 0; i < len; ++i)
        {
            va[i] = mont.toMont(va[i]);
            vb[i] = mont.toMont(vb[i]);
        }

        int root = 3;
        auto start = chrono::high_resolution_clock::now();
        vector<long long> cr = gpu_get_result(va, vb, p_, root, mont);
        auto end = chrono::high_resolution_clock::now();
        ans = chrono::duration<double, ratio<1, 1000>>(end - start).count();

        for (int i = 0; i < 2 * n_ - 1; ++i)
        {
            ab[i] = (int)mont.fromMont(cr[i]);
        }

        fCheck(ab, n_, id);
        cout << "GPU average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
        fWrite(ab, n_, id);
    }
    return 0;
}