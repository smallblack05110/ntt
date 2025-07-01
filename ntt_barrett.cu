#include <cuda_runtime.h>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// CUDA错误检查宏
#define CHECK_CUDA(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
             << " - " << cudaGetErrorString(error) << endl; \
        exit(1); \
    } \
} while(0)

// 巴雷特模乘 - 修正版
__device__ uint64_t barrett_reduce(uint64_t a, uint64_t b, uint64_t mod, uint64_t mu) {
    __uint128_t product = (__uint128_t)a * b;
    uint64_t q = (uint64_t)((product * (__uint128_t)mu) >> 64);
    uint64_t r = (uint64_t)(product - q * mod);
    return r < mod ? r : r - mod;
}

// 快速幂（GPU版本）- 使用巴雷特模乘
__device__ uint64_t quick_pow_gpu(uint64_t base, uint64_t exp, uint64_t mod, uint64_t mu) {
    uint64_t result = 1;
    base = barrett_reduce(base, 1, mod, mu); // 确保base在范围内
    while (exp > 0) {
        if (exp & 1) {
            result = barrett_reduce(result, base, mod, mu);
        }
        base = barrett_reduce(base, base, mod, mu);
        exp >>= 1;
    }
    return result;
}

// 位反转kernel - 修正版
__global__ void bit_reverse_kernel(uint64_t* data, int n, int log2n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    unsigned int j = 0;
    unsigned int i = idx;
    for (int k = 0; k < log2n; k++) {
        j = (j << 1) | (i & 1);
        i >>= 1;
    }
    
    if (idx < j) {
        uint64_t tmp = data[idx];
        data[idx] = data[j];
        data[j] = tmp;
    }
}

// NTT核心kernel - 修正版
__global__ void ntt_kernel(uint64_t* data, int n, int len, uint64_t wn, uint64_t p, uint64_t mu, bool inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_len = len / 2;
    int num_butterflies = n / len;
    
    if (idx >= num_butterflies * half_len) return;
    
    int butterfly_group = idx / half_len;
    int butterfly_idx = idx % half_len;
    int i = butterfly_group * len + butterfly_idx;
    int j = i + half_len;
    
    uint64_t w = quick_pow_gpu(wn, (uint64_t)butterfly_idx, p, mu);
    
    uint64_t u = data[i];
    uint64_t v = barrett_reduce(data[j], w, p, mu);
    
    data[i] = (u + v) % p;
    data[j] = (u + p - v) % p;
}

// 点乘kernel - 修正版
__global__ void pointwise_multiply_kernel(uint64_t* a, uint64_t* b, uint64_t* c, int n, uint64_t p, uint64_t mu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    c[idx] = barrett_reduce(a[idx], b[idx], p, mu);
}

// 标量乘法kernel - 修正版
__global__ void scalar_multiply_kernel(uint64_t* data, int n, uint64_t scalar, uint64_t p, uint64_t mu) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    data[idx] = barrett_reduce(data[idx], scalar, p, mu);
}

// 主机端快速幂
uint64_t quick_pow_host(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp & 1) {
            result = ((__uint128_t)result * base) % mod;
        }
        base = ((__uint128_t)base * base) % mod;
        exp >>= 1;
    }
    return result;
}

// 计算巴雷特常数mu
uint64_t calculate_mu(uint64_t p) {
    __uint128_t mu = (__uint128_t(1) << 64) / p;
    return uint64_t(mu);
}

// 检查是否为2的幂
bool is_power_of_two(int n) {
    return n > 0 && (n & (n - 1)) == 0;
}

// GPU上的NTT实现 - 修正版
void ntt_gpu(uint64_t* d_data, int n, uint64_t p, uint64_t root, uint64_t mu, bool inverse) {
    if (!is_power_of_two(n)) {
        cerr << "Error: n must be a power of 2" << endl;
        return;
    }
    
    const int THREADS_PER_BLOCK = 256;
    int log2n = 0;
    int temp = n;
    while (temp >>= 1) log2n++;
    
    // 位反转
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    bit_reverse_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, n, log2n);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Cooley-Tukey NTT
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t wn = quick_pow_host(root, (p - 1) / len, p);
        if (inverse) {
            wn = quick_pow_host(wn, p - 2, p);
        }
        
        int num_operations = n / 2;
        blocks = (num_operations + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        ntt_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, n, len, wn, p, mu, inverse);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    
    if (inverse) {
        uint64_t inv_n = quick_pow_host(n, p - 2, p);
        blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        scalar_multiply_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_data, n, inv_n, p, mu);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
    }
}

// 文件读取
void fRead(uint64_t *a, uint64_t *b, int *n, int64_t *p, int input_id) {
    string filename = "./nttdata/" + to_string(input_id) + ".in";
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Cannot open input file: " << filename << endl;
        exit(1);
    }
    fin >> *n >> *p;
    for (int i = 0; i < *n; ++i) {
        fin >> a[i];
    }
    for (int i = 0; i < *n; ++i) {
        fin >> b[i];
    }
    fin.close();
}

// 文件写入
void fWrite(const uint64_t *ab, int n, int input_id) {
    string filename = "files/" + to_string(input_id) + ".out";
    ofstream fout(filename);
    for (int i = 0; i < n * 2 - 1; ++i) {
        fout << ab[i] << '\n';
    }
    fout.close();
}

// 结果检查
void fCheck(uint64_t *ab, int n, int input_id) {
    string filename = "./nttdata/" + to_string(input_id) + ".out";
    ifstream fin(filename);
    if (!fin.is_open()) {
        cerr << "Cannot open check file: " << filename << endl;
        return;
    }
    
    bool correct = true;
    for (int i = 0; i < n * 2 - 1; i++) {
        uint64_t x;
        fin >> x;
        if (x != ab[i]) {
            correct = false;
            cout << "Error at position " << i << ": expected " << x << ", got " << ab[i] << endl;
            // 只打印第一个错误
            break;
        }
    }
    
    if (correct) {
        cout << "多项式乘法结果正确" << endl;
    } else {
        cout << "多项式乘法结果错误" << endl;
    }
    fin.close();
}

int main(int argc, char *argv[]) {
    int test_begin = 4, test_end = 0;
    const uint64_t root = 3;  // 原始单位根
    
    // 分配最大可能需要的主机内存
    const int MAX_SIZE = 300000;
    uint64_t *h_a = new uint64_t[MAX_SIZE];
    uint64_t *h_b = new uint64_t[MAX_SIZE];
    uint64_t *h_c = new uint64_t[MAX_SIZE];
    
    for (int id = test_begin; id >= test_end; --id) {
        int n;
        int64_t p;
        
        // 读取输入
        fRead(h_a, h_b, &n, &p, id);
        cout << "Processing test " << id << ": n = " << n << ", p = " << p << endl;
        
        // 计算需要的长度（2的幂）
        int len = 1;
        while (len < 2 * n) len <<= 1;
        cout << "Using FFT length: " << len << endl;
        
        // 填充零
        fill(h_a + n, h_a + len, 0);
        fill(h_b + n, h_b + len, 0);
        fill(h_c, h_c + len, 0);
        
        // 预计算巴雷特常数mu
        uint64_t mu = calculate_mu(p);
        
        // 分配GPU内存
        uint64_t *d_a, *d_b, *d_c;
        CHECK_CUDA(cudaMalloc(&d_a, len * sizeof(uint64_t)));
        CHECK_CUDA(cudaMalloc(&d_b, len * sizeof(uint64_t)));
        CHECK_CUDA(cudaMalloc(&d_c, len * sizeof(uint64_t)));
        
        // 复制数据到GPU
        CHECK_CUDA(cudaMemcpy(d_a, h_a, len * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_b, h_b, len * sizeof(uint64_t), cudaMemcpyHostToDevice));
        
        auto start = chrono::high_resolution_clock::now();
        
        // 执行前向NTT
        ntt_gpu(d_a, len, p, root, mu, false);
        ntt_gpu(d_b, len, p, root, mu, false);
        
        // 点乘
        const int THREADS_PER_BLOCK = 256;
        int blocks = (len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        pointwise_multiply_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, len, p, mu);
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        
        // 执行逆NTT
        ntt_gpu(d_c, len, p, root, mu, true);
        
        auto end = chrono::high_resolution_clock::now();
        
        // 复制结果回主机
        CHECK_CUDA(cudaMemcpy(h_c, d_c, len * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        
        // 计算时间
        double time_ms = chrono::duration<double, milli>(end - start).count();
        
        cout << "GPU latency for n = " << n << " p = " << p << " : " << time_ms << " us" << endl;
        
        // 检查结果
        fCheck(h_c, n, id);
        
        // 写入结果
        fWrite(h_c, n, id);
        
        // 释放GPU内存
        CHECK_CUDA(cudaFree(d_a));
        CHECK_CUDA(cudaFree(d_b));
        CHECK_CUDA(cudaFree(d_c));
    }
    
    // 释放主机内存
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return 0;
}