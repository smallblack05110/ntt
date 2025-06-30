#include <cuda_runtime.h>
#include <cuda.h>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n", \
                    __FILE__, __LINE__, error, cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// 常量内存
__constant__ long long d_MOD;

// Host上的快速幂取模
long long host_pow_mod(long long a, long long b, long long p) {
    long long result = 1;
    a %= p;
    while (b > 0) {
        if (b & 1) {
            result = (__int128)result * a % p;
        }
        a = (__int128)a * a % p;
        b >>= 1;
    }
    return result;
}

// GPU上的快速幂取模
__device__ long long gpu_pow_mod(long long a, long long b, long long p) {
    long long result = 1;
    a %= p;
    while (b > 0) {
        if (b & 1) {
            result = (__int128)result * a % p;
        }
        a = (__int128)a * a % p;
        b >>= 1;
    }
    return result;
}

// 位逆序
__device__ __host__ int reverse_bits(int x, int log_n) {
    int result = 0;
    for (int i = 0; i < log_n; i++) {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    return result;
}

// 位逆序重排核函数
__global__ void bit_reverse_permute(long long* data, int n, int log_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int rev_idx = reverse_bits(idx, log_n);
        if (idx < rev_idx) {
            long long temp = data[idx];
            data[idx] = data[rev_idx];
            data[rev_idx] = temp;
        }
    }
}

// 正确的NTT蝶形运算核函数
__global__ void ntt_butterfly_kernel(long long* data, int n, int len, long long wn, bool invert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_butterflies = n / 2;
    
    if (idx < total_butterflies) {
        int i = (idx / (len / 2)) * len + (idx % (len / 2));
        int j = i + len / 2;
        
        // 计算旋转因子w^(idx % (len/2))
        long long w = 1;
        int exp = idx % (len / 2);
        if (exp != 0) {
            w = gpu_pow_mod(wn, exp, d_MOD);
            if (invert) {
                w = gpu_pow_mod(w, d_MOD - 2, d_MOD);  // 求逆元
            }
        }
        
        long long u = data[i];
        long long v = (__int128)data[j] * w % d_MOD;
        
        data[i] = (u + v) % d_MOD;
        data[j] = (u - v + d_MOD) % d_MOD;
    }
}

// 标量乘法核函数
__global__ void scalar_multiply(long long* data, long long scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = (__int128)data[idx] * scalar % d_MOD;
    }
}

// 点乘核函数
__global__ void pointwise_multiply(long long* a, long long* b, long long* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = (__int128)a[idx] * b[idx] % d_MOD;
    }
}

// 检查是否为原根的辅助函数
bool is_primitive_root(long long g, long long p) {
    if (g <= 1 || g >= p) return false;
    
    // 检查g^((p-1)/q) != 1 (mod p)，对于所有质因数q
    long long phi = p - 1;
    vector<long long> factors;
    
    // 简单的因数分解（对于小的质因数）
    for (long long i = 2; i * i <= phi; i++) {
        if (phi % i == 0) {
            factors.push_back(i);
            while (phi % i == 0) phi /= i;
        }
    }
    if (phi > 1) factors.push_back(phi);
    
    phi = p - 1;
    for (long long factor : factors) {
        if (host_pow_mod(g, phi / factor, p) == 1) {
            return false;
        }
    }
    return true;
}

// 找到原根
long long find_primitive_root(long long p) {
    for (long long g = 2; g < p; g++) {
        if (is_primitive_root(g, p)) {
            return g;
        }
    }
    return 3; // 默认返回3
}

// 优化的GPU NTT类
class OptimizedGpuNTT {
private:
    long long mod;
    long long root;
    long long* d_buffer1;
    long long* d_buffer2;
    long long* d_buffer3;
    int max_size;
    cudaStream_t stream;
    
public:
    OptimizedGpuNTT(long long modulus, int max_n = 1 << 20) 
        : mod(modulus), max_size(max_n) {
        
        // 自动找到原根
        root = find_primitive_root(mod);
        cout << "使用原根: " << root << " 对于模数: " << mod << endl;
        
        // 设置常量内存
        CUDA_CHECK(cudaMemcpyToSymbol(d_MOD, &mod, sizeof(long long)));
        
        // 分配GPU内存缓冲区
        size_t buffer_size = max_size * sizeof(long long);
        CUDA_CHECK(cudaMalloc(&d_buffer1, buffer_size));
        CUDA_CHECK(cudaMalloc(&d_buffer2, buffer_size));
        CUDA_CHECK(cudaMalloc(&d_buffer3, buffer_size));
        
        // 创建CUDA流
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    ~OptimizedGpuNTT() {
        CUDA_CHECK(cudaFree(d_buffer1));
        CUDA_CHECK(cudaFree(d_buffer2));
        CUDA_CHECK(cudaFree(d_buffer3));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }
    
    void ntt_transform(long long* d_data, int n, bool invert = false) {
        int log_n = 0;
        int temp = n;
        while (temp > 1) {
            log_n++;
            temp >>= 1;
        }
        
        // 位逆序排列
        int block_size = min(256, n);
        int grid_size = (n + block_size - 1) / block_size;
        bit_reverse_permute<<<grid_size, block_size, 0, stream>>>(d_data, n, log_n);
        
        // NTT蝶形运算
        for (int len = 2; len <= n; len <<= 1) {
            // 计算本轮的n次单位根
            long long wn = host_pow_mod(root, (mod - 1) / len, mod);
            
            int butterflies = n / 2;
            int opt_block_size = min(256, butterflies);
            int opt_grid_size = (butterflies + opt_block_size - 1) / opt_block_size;
            
            ntt_butterfly_kernel<<<opt_grid_size, opt_block_size, 0, stream>>>(
                d_data, n, len, wn, invert
            );
        }
        
        // 如果是逆变换，乘以n的逆元
        if (invert) {
            long long inv_n = host_pow_mod(n, mod - 2, mod);
            scalar_multiply<<<grid_size, block_size, 0, stream>>>(d_data, inv_n, n);
        }
    }
    
    void convolution(long long* h_a, long long* h_b, long long* h_result, int n) {
        // 计算所需长度
        int len = 1;
        while (len < 2 * n) len <<= 1;
        
        size_t copy_size = n * sizeof(long long);
        
        // 异步复制数据到GPU并清零填充部分
        CUDA_CHECK(cudaMemcpyAsync(d_buffer1, h_a, copy_size, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_buffer2, h_b, copy_size, cudaMemcpyHostToDevice, stream));
        
        if (len > n) {
            CUDA_CHECK(cudaMemsetAsync(d_buffer1 + n, 0, (len - n) * sizeof(long long), stream));
            CUDA_CHECK(cudaMemsetAsync(d_buffer2 + n, 0, (len - n) * sizeof(long long), stream));
        }
        
        // 前向NTT
        ntt_transform(d_buffer1, len, false);
        ntt_transform(d_buffer2, len, false);
        
        // 点乘
        int block_size = 256;
        int grid_size = (len + block_size - 1) / block_size;
        pointwise_multiply<<<grid_size, block_size, 0, stream>>>(
            d_buffer1, d_buffer2, d_buffer3, len
        );
        
        // 逆NTT
        ntt_transform(d_buffer3, len, true);
        
        // 复制结果回主机
        size_t result_size = (2 * n - 1) * sizeof(long long);
        CUDA_CHECK(cudaMemcpyAsync(h_result, d_buffer3, result_size, cudaMemcpyDeviceToHost, stream));
        
        // 等待所有操作完成
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
};

// 辅助函数
void fRead(int *a, int *b, int *n, int *p, int input_id) {
    string str1 = "./nttdata/";
    string str2 = to_string(input_id);
    string strin = str1 + str2 + ".in";
    char data_path[256];
    strncpy(data_path, strin.c_str(), sizeof(data_path));
    data_path[sizeof(data_path) - 1] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    if (!fin) {
        cerr << "无法打开输入文件: " << strin << endl;
        return;
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

void fCheck(long long *ab, int n, int input_id) {
    string str1 = "./nttdata/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char data_path[256];
    strncpy(data_path, strout.c_str(), sizeof(data_path));
    data_path[sizeof(data_path) - 1] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    if (!fin) {
        cerr << "无法打开输出文件: " << strout << endl;
        return;
    }
    
    bool correct = true;
    for (int i = 0; i < n * 2 - 1; ++i) {
        long long x;
        fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误在位置 " << i << ": 期望 " << x << ", 得到 " << ab[i] << endl;
            correct = false;
            break;
        }
    }
    if (correct) {
        cout << "多项式乘法结果正确" << endl;
    }
    fin.close();
}

void fWrite(long long *ab, int n, int input_id) {
    string str1 = "files/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char output_path[256];
    strncpy(output_path, strout.c_str(), sizeof(output_path));
    output_path[sizeof(output_path) - 1] = '\0';
    ofstream fout;
    fout.open(output_path, ios::out);
    if (!fout) {
        cerr << "无法打开输出文件用于写入: " << strout << endl;
        return;
    }
    for (int i = 0; i < n * 2 - 1; ++i) {
        fout << ab[i] << '\n';
    }
    fout.close();
}

// 全局数组
int a[300000], b[300000];
long long ab[300000];

int main(int argc, char *argv[]) {
    // GPU信息查询
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    cout << "发现 " << device_count << " 个CUDA设备" << endl;
    
    if (device_count == 0) {
        cerr << "没有发现CUDA设备！" << endl;
        return 1;
    }
    
    // 设置GPU参数以提高性能
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    cout << "\n使用GPU: " << prop.name << endl;
    cout << "计算能力: " << prop.major << "." << prop.minor << endl;
    cout << "SM数量: " << prop.multiProcessorCount << endl;
    cout << endl;
    
    int test_begin = 0, test_end = 3;
    
    for (int id = test_begin; id <= test_end; ++id) {
        int n_, p_;
        
        // 读取输入
        fRead(a, b, &n_, &p_, id);
        memset(ab, 0, sizeof(ab));
        
        // 创建优化的NTT实例
        OptimizedGpuNTT ntt_engine(p_);
        
        // 转换输入数据类型
        vector<long long> va(n_), vb(n_);
        for (int i = 0; i < n_; i++) {
            va[i] = a[i];
            vb[i] = b[i];
        }
        
        // 单次运行测试正确性
        ntt_engine.convolution(va.data(), vb.data(), ab, n_);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 验证结果
        fCheck(ab, n_, id);
        
        // 性能测试
        auto start = chrono::high_resolution_clock::now();
        
        for (int iter = 0; iter < 10; iter++) {
            // 重新准备数据
            for (int i = 0; i < n_; i++) {
                va[i] = a[i];
                vb[i] = b[i];
            }
            ntt_engine.convolution(va.data(), vb.data(), ab, n_);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        auto end = chrono::high_resolution_clock::now();
        double avg_time = chrono::duration<double, std::milli>(end - start).count() / 10.0;
        
        cout << "average latency for n = " << n_ << " p = " << p_ << " : " 
             << fixed << setprecision(4) << avg_time << " us" << endl;
        
        // 写入结果
        fWrite(ab, n_, id);
    }
    
    return 0;
}