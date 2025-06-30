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

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

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

int quick_mod(int a, int b, int p) {
    int result = 1;
    a = a % p;
    while (b > 0) {
        if (b % 2 == 1) {
            result = (1LL * result * a) % p;
        }
        a = (1LL * a * a) % p;
        b /= 2;
    }
    return result;
}

__device__ int gpu_quick_mod(long long a, int b, int p) {
    long long result = 1;
    a = a % p;
    while (b > 0) {
        if (b % 2 == 1) {
            result = (result * a) % p;
        }
        a = (a * a) % p;
        b /= 2;
    }
    return (int)result;
}

// 修复的位逆序kernel
__global__ void bit_reverse_kernel(int *a, int n, int log_n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    // 正确的位逆序算法
    int j = 0;
    for (int i = 0; i < log_n; i++) {
        if (idx & (1 << i)) {
            j |= 1 << (log_n - 1 - i);
        }
    }
    
    // 只有当idx < j时才交换，避免重复交换
    if (idx < j) {
        int temp = a[idx];
        a[idx] = a[j];
        a[j] = temp;
    }
}

// 优化的NTT变换kernel - 使用更好的并行策略
__global__ void ntt_transform_kernel(int *a, int n, int len, int p, int wn, bool invert) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    // 处理所有长度为len的蝶形操作
    for (int i = idx; i < n; i += stride) {
        int group = i / len;           // 当前组号
        int pos_in_group = i % len;    // 在组内的位置
        
        if (pos_in_group >= len / 2) continue;  // 只处理前半部分
        
        int base = group * len;
        int j = pos_in_group;
        int k = j + len / 2;
        
        // 计算旋转因子 w = wn^j
        long long w = 1;
        int exp = j;
        long long base_w = wn;
        while (exp > 0) {
            if (exp & 1) w = (w * base_w) % p;
            base_w = (base_w * base_w) % p;
            exp >>= 1;
        }
        
        int op1 = a[base + j];
        long long op2 = ((long long)a[base + k] * w) % p;
        
        a[base + j] = (op1 + op2) % p;
        a[base + k] = (op1 - op2 + p) % p;
    }
}

// 点乘kernel
__global__ void pointwise_mul_kernel(int *a, int *b, int *c, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = ((long long)a[idx] * b[idx]) % p;
    }
}

// 标量乘法kernel
__global__ void scalar_mul_kernel(int *a, int inv_n, int n, int p) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = ((long long)a[idx] * inv_n) % p;
    }
}

class GPU_NTT {
private:
    int *d_a, *d_b, *d_c;
    int max_size;
    
public:
    GPU_NTT(int _max_size) : max_size(_max_size) {
        CUDA_CHECK(cudaMalloc(&d_a, max_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_b, max_size * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_c, max_size * sizeof(int)));
    }
    
    ~GPU_NTT() {
        if (d_a) cudaFree(d_a);
        if (d_b) cudaFree(d_b);
        if (d_c) cudaFree(d_c);
    }
    
    void ntt_iter_gpu(int *d_data, int n, int p, int root, bool invert) {
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
            int wn = quick_mod(root, (p - 1) / len, p);
            if (invert) {
                wn = quick_mod(wn, p - 2, p);
            }
            
            // 优化并行度：确保有足够的工作给GPU
            int total_ops = n / 2;  // 每层的蝶形操作总数
            int threads = min(1024, total_ops);
            int blocks_needed = min((total_ops + threads - 1) / threads, 
                                  gpu_info.sm_count * 4);
            
            ntt_transform_kernel<<<blocks_needed, threads>>>(d_data, n, len, p, wn, invert);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
    vector<int> get_result_gpu(vector<int> &a_vec, vector<int> &b_vec, int p, int len, int root) {
        // 一次性将所有数据传输到GPU
        CUDA_CHECK(cudaMemcpy(d_a, a_vec.data(), len * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_b, b_vec.data(), len * sizeof(int), cudaMemcpyHostToDevice));
        
        // 在GPU上完成所有计算
        ntt_iter_gpu(d_a, len, p, root, false);  // 前向NTT for a
        ntt_iter_gpu(d_b, len, p, root, false);  // 前向NTT for b
        
        // 点乘
        int threads_per_block = min(1024, gpu_info.max_threads_per_block);
        int blocks = (len + threads_per_block - 1) / threads_per_block;
        pointwise_mul_kernel<<<blocks, threads_per_block>>>(d_a, d_b, d_c, len, p);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 逆NTT
        ntt_iter_gpu(d_c, len, p, root, true);
        
        // 乘以逆元
        int inv_n = quick_mod(len, p - 2, p);
        scalar_mul_kernel<<<blocks, threads_per_block>>>(d_c, inv_n, len, p);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 一次性传输结果回CPU
        vector<int> c_vec(len);
        CUDA_CHECK(cudaMemcpy(c_vec.data(), d_c, len * sizeof(int), cudaMemcpyDeviceToHost));
        
        return c_vec;
    }
};

// 原有的CPU版本作为参考和验证
void ntt_iter_cpu(vector<int>& a, int p, int root, bool invert) {
    int n = a.size();
    int half = n / 2;
    for (int i = 1, j = 0; i < n; i++) { 
         int bit = half;
        for (; j >= bit; bit /= 2) {
            j -= bit;
        }
        j += bit;
        if (i < j) {
            swap(a[i], a[j]);
        }
    }
    
    for (int len = 2; len <= n; len *= 2) {
        int wn;
        wn = quick_mod(root, (p - 1) / len, p);
        if(invert) {
            wn = quick_mod(wn, p - 2, p);
        }
        for (int i = 0; i < n; i += len) {
            int w0 = 1;
            for (int j = 0; j < len / 2; j++) {
                int op1 = a[i + j];
                int op2 = (1LL * a[i + j + len / 2] * w0) % p;
                a[i + j] = (op1 + op2) % p;
                a[i + j + len / 2] = (op1 - op2 + p) % p;
                w0 = (1LL * w0 * wn) % p;
            }
        }
    }
}

vector<int> get_result_cpu(vector<int> &a_vec, vector<int> &b_vec, int p, int len, int root) {
    ntt_iter_cpu(a_vec, p, root, false);
    ntt_iter_cpu(b_vec, p, root, false);
    vector<int> c_vec(len);
    for (int j = 0; j < len; ++j) {
        c_vec[j] = (1LL * a_vec[j] * b_vec[j]) % p;
    }
    ntt_iter_cpu(c_vec, p, root, true);
    int inv_m = quick_mod(len, p - 2, p);
    for (int j = 0; j < len; ++j) {
        c_vec[j] = (1LL * c_vec[j] * inv_m) % p;
    }
    return c_vec;
}

// 文件I/O函数保持不变
void fRead(int *a, int *b, int *n, int *p, int input_id) {
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

void fCheck(int *ab, int n, int input_id) {
    string str1 = "./nttdata/";
    string str2 = to_string(input_id);
    string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    ifstream fin;
    fin.open(data_path, ios::in);
    for (int i = 0; i < n * 2 - 1; ++i) {
        int x;
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

void fWrite(int *ab, int n, int input_id) {
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

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[]) {
    CUDA_CHECK(cudaSetDevice(0));
    gpu_info.init();
    
    GPU_NTT gpu_ntt(300000);
    
    int test_begin = 0;
    int test_end = 4;
    
    for (int i = test_begin; i <= test_end; ++i) {
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, i);

        int len = 1;
        while (len < 2 * n_) {
            len <<= 1;
        }

        fill(a + n_, a + len, 0);
        fill(b + n_, b + len, 0);
        vector<int> a_vec(a, a + len);
        vector<int> b_vec(b, b + len);
        int root = 3;
        
        auto Start = chrono::high_resolution_clock::now();
        
        // 使用修复后的GPU版本
        vector<int> c_vec = gpu_ntt.get_result_gpu(a_vec, b_vec, p_, len, root);
        
        // 可选：与CPU版本比较验证正确性
        // vector<int> a_vec_cpu = a_vec, b_vec_cpu = b_vec;
        // vector<int> c_vec_cpu = get_result_cpu(a_vec_cpu, b_vec_cpu, p_, len, root);
        
        for (int j = 0; j < 2 * n_ - 1; ++j) {
            ab[j] = c_vec[j];
        }

        auto End = chrono::high_resolution_clock::now();
        chrono::duration<double, ratio<1, 1000>> elapsed = End - Start;
        ans += elapsed.count();

        fCheck(ab, n_, i);
        cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " us" << endl;
        fWrite(ab, n_, i);
    }
    return 0;
}