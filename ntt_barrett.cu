#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <chrono>
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

using namespace std;

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// GPU设备信息
struct GPUInfo {
    int sm_count;
    int max_threads_per_block;
    int warp_size;
    
    void init() {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        sm_count = prop.multiProcessorCount;
        max_threads_per_block = prop.maxThreadsPerBlock;
        warp_size = prop.warpSize;
        printf("GPU: SM Count=%d, Max Threads/Block=%d\n", sm_count, max_threads_per_block);
    }
} gpu_info;

// CPU快速幂
uint64_t quick_mod(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t result = 1;
    a = a % p;
    while (b > 0) {
        if (b % 2 == 1) {
            result = (__uint128_t(result) * a) % p;
        }
        a = (__uint128_t(a) * a) % p;
        b /= 2;
    }
    return result;
}

// GPU快速幂
__device__ uint64_t gpu_quick_mod(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t result = 1;
    a = a % p;
    while (b > 0) {
        if (b % 2 == 1) {
            result = ((__uint128_t)result * a) % p;
        }
        a = ((__uint128_t)a * a) % p;
        b /= 2;
    }
    return result;
}

// GPU模乘
__device__ uint64_t gpu_mod_mul(uint64_t a, uint64_t b, uint64_t p) {
    return ((__uint128_t)a * b) % p;
}

// 检查模数是否支持NTT
bool check_ntt_support(uint64_t p, int required_len) {
    if (p <= 1) return false;
    
    // 计算需要的2的幂次
    int required_power = 0;
    int len = required_len;
    while (len > 1) {
        len /= 2;
        required_power++;
    }
    
    // 检查p-1中2的幂次
    uint64_t temp = p - 1;
    int power_of_2 = 0;
    while (temp % 2 == 0) {
        temp /= 2;
        power_of_2++;
    }
    
    return power_of_2 >= required_power;
}

// 查找原根
uint64_t find_primitive_root(uint64_t p) {
    // 对于NTT模数，通常3是原根
    if (p == 7340033 || p == 104857601 || p == 469762049 || p == 998244353 || 
        p == 1004535809 || p == 1224736769) {
        return 3;
    }
    
    // 注意：2147483647 不支持大长度NTT，从列表中移除
    
    // 简单的原根查找（仅适用于小素数）
    for (uint64_t g = 2; g < p && g < 100; g++) {  // 限制搜索范围避免超时
        uint64_t order = 1;
        uint64_t temp = g;
        while (temp != 1 && order < p) {
            temp = ((__uint128_t)temp * g) % p;
            order++;
            if (order > p - 1) break;
        }
        if (order == p - 1) return g;
    }
    
    // 默认返回3
    return 3;
}

// 位逆序kernel
__global__ void bit_reverse_kernel(uint64_t *a, int n, int log_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    int j = 0;
    for (int i = 0; i < log_n; i++) {
        if (idx & (1 << i)) {
            j |= 1 << (log_n - 1 - i);
        }
    }
    
    if (idx < j) {
        uint64_t temp = a[idx];
        a[idx] = a[j];
        a[j] = temp;
    }
}

// NTT变换kernel
__global__ void ntt_transform_kernel(uint64_t *a, int n, int len, uint64_t p, uint64_t wn) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = idx; i < n; i += stride) {
        int group = i / len;
        int pos_in_group = i % len;
        
        if (pos_in_group >= len / 2) continue;
        
        int base = group * len;
        int j = pos_in_group;
        int k = j + len / 2;
        
        // 计算旋转因子 w = wn^j
        uint64_t w = 1;
        int exp = j;
        uint64_t base_w = wn;
        while (exp > 0) {
            if (exp & 1) w = gpu_mod_mul(w, base_w, p);
            base_w = gpu_mod_mul(base_w, base_w, p);
            exp >>= 1;
        }
        
        uint64_t op1 = a[base + j];
        uint64_t op2 = gpu_mod_mul(a[base + k], w, p);
        
        a[base + j] = (op1 + op2) % p;
        a[base + k] = (op1 - op2 + p) % p;
    }
}

// 点乘kernel
__global__ void pointwise_mul_kernel(uint64_t *a, uint64_t *b, uint64_t *c, int n, uint64_t p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = gpu_mod_mul(a[idx], b[idx], p);
    }
}

// 标量乘法kernel
__global__ void scalar_mul_kernel(uint64_t *a, uint64_t inv_n, int n, uint64_t p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = gpu_mod_mul(a[idx], inv_n, p);
    }
}

class GPU_NTT {
private:
    uint64_t *d_a, *d_b, *d_c;
    int max_size;
    
public:
    GPU_NTT(int _max_size) : max_size(_max_size) {
        CUDA_CHECK(cudaMalloc(&d_a, max_size * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_b, max_size * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_c, max_size * sizeof(uint64_t)));
    }
    
    ~GPU_NTT() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
    }
    
    void ntt_iter_gpu(uint64_t *d_data, int n, uint64_t p, uint64_t root, bool invert) {
        // 计算log_n
        int log_n = 0;
        int temp = n;
        while (temp > 1) {
            temp >>= 1;
            log_n++;
        }
        
        // 位逆序
        int threads_per_block = min(1024, gpu_info.max_threads_per_block);
        int blocks = (n + threads_per_block - 1) / threads_per_block;
        bit_reverse_kernel<<<blocks, threads_per_block>>>(d_data, n, log_n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // NTT变换
        for (int len = 2; len <= n; len *= 2) {
            uint64_t wn = quick_mod(root, (p - 1) / len, p);
            if (invert) {
                wn = quick_mod(wn, p - 2, p);
            }
            
            int total_ops = n / 2;
            int threads = min(1024, total_ops);
            int blocks_needed = min((total_ops + threads - 1) / threads, 
                                  gpu_info.sm_count * 4);
            
            ntt_transform_kernel<<<blocks_needed, threads>>>(d_data, n, len, p, wn);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    vector<uint64_t> get_result_gpu(vector<uint64_t> &a_vec, vector<uint64_t> &b_vec, 
                                   uint64_t p, int len, uint64_t root) {
        // 传输数据到GPU
        CUDA_CHECK(cudaMemcpy(d_a, a_vec.data(), len * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b_vec.data(), len * sizeof(uint64_t), cudaMemcpyHostToDevice));
        
        // 前向NTT
        ntt_iter_gpu(d_a, len, p, root, false);
        ntt_iter_gpu(d_b, len, p, root, false);
        
        // 点乘
        int threads_per_block = min(1024, gpu_info.max_threads_per_block);
        int blocks = (len + threads_per_block - 1) / threads_per_block;
        pointwise_mul_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, len, p);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 逆NTT
        ntt_iter_gpu(d_c, len, p, root, true);
        
        // 乘以逆元
        uint64_t inv_n = quick_mod(len, p - 2, p);
        scalar_mul_kernel<<<blocks, threads_per_block>>>(d_c, inv_n, len, p);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 传输结果回CPU
        vector<uint64_t> c_vec(len);
        CUDA_CHECK(cudaMemcpy(c_vec.data(), d_c, len * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        
        return c_vec;
    }
};

// CPU版本作为备用和验证
void ntt_iter_cpu(vector<uint64_t> &a, uint64_t p, uint64_t root, bool invert) {
    int n = a.size();

    // 位反转置换
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;
        if (i < j)
            swap(a[i], a[j]);
    }

    // NTT主循环
    for (int len = 2; len <= n; len <<= 1) {
        uint64_t wn = quick_mod(root, (p - 1) / len, p);
        if (invert)
            wn = quick_mod(wn, p - 2, p);

        for (int i = 0; i < n; i += len) {
            uint64_t w = 1;
            for (int j = 0; j < len / 2; ++j) {
                uint64_t u = a[i + j];
                uint64_t v = ((__uint128_t)w * a[i + j + len / 2]) % p;
                a[i + j] = (u + v) % p;
                a[i + j + len / 2] = (u - v + p) % p;
                w = ((__uint128_t)w * wn) % p;
            }
        }
    }
}

vector<uint64_t> get_result_cpu(vector<uint64_t> &a, vector<uint64_t> &b, uint64_t p, uint64_t root) {
    int n = a.size();
    ntt_iter_cpu(a, p, root, false);
    ntt_iter_cpu(b, p, root, false);
    vector<uint64_t> c(n);
    for (int i = 0; i < n; ++i)
        c[i] = ((__uint128_t)a[i] * b[i]) % p;
    ntt_iter_cpu(c, p, root, true);
    uint64_t inv_n = quick_mod(n, p - 2, p);
    for (int i = 0; i < n; ++i)
        c[i] = ((__uint128_t)c[i] * inv_n) % p;
    return c;
}

// 文件I/O函数
void fRead(uint64_t *a, uint64_t *b, int *n, int64_t *p, int input_id) {
    string str1 = "./nttdata/";
    string str2 = to_string(input_id);
    string strin = str1 + str2 + ".in";
    char data_path[strin.size() + 1];
    copy(strin.begin(), strin.end(), data_path);
    data_path[strin.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    cout << "Reading input from: " << data_path << endl;
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
    string str1 = "files/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char output_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), output_path);
    output_path[strout.size()] = '\0';
    ofstream fout;
    fout.open(output_path, ios::out);
    for (int i = 0; i < n * 2 - 1; ++i) {
        fout << ab[i] << '\n';
    }
    fout.close();
}

void fCheck(uint64_t *ab, int n, int input_id) {
    string str1 = "./nttdata/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    for (int i = 0; i < n * 2 - 1; i++) {
        uint64_t x;
        fin >> x;
        if (x != ab[i]) {
            cout << "多项式乘法结果错误，位置 " << i << ": 期望 " << x << ", 实际 " << ab[i] << endl;
            fin.close();
            return;
        }
    }
    cout << "多项式乘法结果正确" << endl;
    fin.close();
}

uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[]) {
    // 初始化CUDA
    CUDA_CHECK(cudaSetDevice(0));
    gpu_info.init();
    
    // 创建GPU NTT对象
    GPU_NTT gpu_ntt(300000);
    
    int test_begin = 0, test_end = 3;
    
    for (int id = test_begin; id <= test_end; ++id) {
        long double ans = 0;
        int n_;
        int64_t p_;
        fRead(a, b, &n_, &p_, id);
        
        int len = 1;
        while (len < 2 * n_) {
            len <<= 1;
        }
        
        fill(a + n_, a + len, 0);
        fill(b + n_, b + len, 0);
        
        // 检查模数是否支持NTT
        uint64_t p_uint = (uint64_t)p_;
        
        auto start = chrono::high_resolution_clock::now();
        
        if (!check_ntt_support(p_uint, len)) {
            cout << "错误: 模数 " << p_uint << " 不支持长度为 " << len << " 的NTT" << endl;
            cout << "需要使用传统多项式乘法或CRT方法" << endl;
            
            // 使用简单的O(n^2)多项式乘法作为后备
            fill(ab, ab + 2 * n_ - 1, 0);
            for (int i = 0; i < n_; i++) {
                for (int j = 0; j < n_; j++) {
                    ab[i + j] = (ab[i + j] + ((__uint128_t)a[i] * b[j]) % p_uint) % p_uint;
                }
            }
            
            auto end = chrono::high_resolution_clock::now();
            ans = chrono::duration<double, ratio<1, 1000>>(end - start).count();
            
            fCheck(ab, n_, id);
            cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " us (naive method)" << endl;
            fWrite(ab, n_, id);
            continue;
        }
        
        uint64_t root = find_primitive_root(p_uint);
        
        vector<uint64_t> a_vec(a, a + len);
        vector<uint64_t> b_vec(b, b + len);
        
        // 使用GPU版本
        vector<uint64_t> c_vec = gpu_ntt.get_result_gpu(a_vec, b_vec, p_uint, len, root);
        
        // 可选：CPU版本验证（小规模时）
        if (len <= 4096) {
            vector<uint64_t> a_vec_cpu = a_vec, b_vec_cpu = b_vec;
            vector<uint64_t> c_vec_cpu = get_result_cpu(a_vec_cpu, b_vec_cpu, p_uint, root);
            
            // 比较结果
            bool match = true;
            for (int i = 0; i < len && match; i++) {
                if (c_vec[i] != c_vec_cpu[i]) {
                    cout << "GPU/CPU结果不匹配，位置 " << i << ": GPU=" << c_vec[i] << ", CPU=" << c_vec_cpu[i] << endl;
                    match = false;
                }
            }
            if (match) {
                cout << "GPU/CPU结果验证通过" << endl;
            }
        }
        
        for (int j = 0; j < 2 * n_ - 1; ++j) {
            ab[j] = c_vec[j];
        }
        
        auto end = chrono::high_resolution_clock::now();
        ans = chrono::duration<double, ratio<1, 1000>>(end - start).count();
        
        fCheck(ab, n_, id);
        cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " ms" << endl;
        fWrite(ab, n_, id);
    }
    return 0;
}