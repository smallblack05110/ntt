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
#include <unistd.h>  // 支持 sysconf 获取 CPU 核心数

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

// 巴雷特模乘结构体 - 保持不变
struct BarrettReduction {
    uint64_t mod;
    uint64_t mu;
    uint64_t shift;

    // 构造函数，预计算mu和shift
    BarrettReduction(uint64_t _mod) : mod(_mod) {
        // 计算适当的位移
        shift = 64;
        mu = (static_cast<__uint128_t>(1) << shift) / mod;
    }

    // 快速计算 a % mod
    inline uint64_t reduce(uint64_t a) const {
        if (a < mod) return a;
        
        uint64_t q = (static_cast<__uint128_t>(a) * mu) >> shift;
        uint64_t r = a - q * mod;
        
        return r < mod ? r : r - mod;
    }

    // 计算 (a * b) % mod，优化乘法后取模
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
  if (value == 0)
  {
    return "0";
  }
  char buffer[40];
  int index = 0;
  while (value > 0)
  {
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

void fWrite(const uint64_t *ab, int n, int input_id)
{
  char path_buffer[256];
  sprintf(path_buffer, "files/%d.out", input_id);
  
  ofstream fout;
  fout.open(path_buffer, std::ios::out);
  for (int i = 0; i < n * 2 - 1; ++i)
  {
    fout << ab[i] << '\n';
  }
  fout.close();
}

void fCheck(uint64_t *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
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
  while (b > 0)
  {
    if (b & 1) res = barrett.reduce(res * a);
    a = barrett.reduce(a * a); b >>= 1;
  }
  return res;
}

// 使用巴雷特模乘的NTT迭代实现
void ntt_iter_barrett(vector<uint32_t> &a, uint64_t p, int root, bool invert, BarrettReduction &barrett)
{
  int n = a.size();
  for (int i = 1, j = 0; i < n; ++i)
  {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j |= bit;
    if (i < j) swap(a[i], a[j]);
  }
  
  for (int len = 2; len <= n; len <<= 1)
  {
    uint64_t wn = quick_mod_barrett(root, (p - 1) / len, p, barrett);
    if (invert) wn = quick_mod_barrett(wn, p - 2, p, barrett);
    
    for (int i = 0; i < n; i += len)
    {
      uint32_t w = 1;
      for (int j = 0; j < len / 2; ++j)
      {
        uint32_t u = a[i + j];
        // 使用巴雷特模乘进行点值乘法
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

// OpenMP版本的NTT计算函数
void ntt_compute_omp(vector<uint32_t> &a_vec, vector<uint32_t> &b_vec, 
                    vector<uint32_t> &result, uint64_t p, int root, 
                    BarrettReduction &barrett) 
{
  vector<uint32_t> ta(a_vec), tb(b_vec);
  
  // 使用巴雷特模乘进行NTT
  ntt_iter_barrett(ta, p, root, false, barrett);
  ntt_iter_barrett(tb, p, root, false, barrett);
  
  // 点乘，使用巴雷特模乘
  vector<uint32_t> c(ta.size());
  for (size_t i = 0; i < ta.size(); ++i)
    c[i] = barrett.mul_mod(ta[i], tb[i]);
  
  // 逆变换，使用巴雷特模乘
  ntt_iter_barrett(c, p, root, true, barrett);
  
  // 乘以 n^{-1}，使用巴雷特模乘
  uint64_t inv_n = quick_mod_barrett(ta.size(), p - 2, p, barrett);
  for (size_t i = 0; i < c.size(); ++i)
    c[i] = barrett.mul_mod(c[i], inv_n);
  
  result = std::move(c);
}

// CRT 模逆 - 使用巴雷特模乘
__uint128_t power_barrett(__uint128_t base, __uint32_t exp, __uint32_t mod, BarrettReduction &barrett)
{
  __uint128_t res = 1; base %= mod;
  while (exp > 0)
  {
    if (exp & 1) res = barrett.reduce(res * base);
    base = barrett.reduce(base * base); exp >>= 1;
  }
  return res;
}

__uint128_t modinv_crt_barrett(__uint128_t a, __uint128_t m, BarrettReduction &barrett)
{
  return power_barrett(a, m - 2, m, barrett);
}

uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
  // 获取可用 CPU 核心数
  int num_threads = omp_get_max_threads();
  
  // 设置OpenMP线程数，也可以通过环境变量OMP_NUM_THREADS控制
  omp_set_num_threads(num_threads);
  
  cout << "CPU核心数/OpenMP线程数: " << num_threads << endl;

  int test_begin = 0, test_end = 4;
  const int root = 3;
  const int CRT_CNT = 4;  // 小模数数量，也是NTT并行计算的线程数

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

  // 为大模数M创建巴雷特计算器（用于CRT）
  BarrettReduction barrett_M(M);

  // 预计算 CRT 常量 K 和 invK
  __uint128_t K[CRT_CNT], invK[CRT_CNT];
  for (int i = 0; i < CRT_CNT; ++i)
  {
    K[i] = M / small_mods[i];
    // 使用巴雷特模乘优化
    invK[i] = modinv_crt_barrett(K[i], small_mods[i], *barrett_mods[i]);
  }

  for (int id = test_begin; id <= test_end; ++id)
  {
    long double ans = 0;
    int n_;
    int64_t p_;
    fRead(a, b, &n_, &p_, id);
    int len = 1; 
    while (len < 2 * n_) len <<= 1;
    fill(a + n_, a + len, 0);
    fill(b + n_, b + len, 0);

    auto start = chr::high_resolution_clock::now();

    // 存储每个小模NTT结果 - 修改为32位
    vector<vector<uint32_t>> mods(CRT_CNT);
    for (int i = 0; i < CRT_CNT; i++) {
        mods[i].resize(len);
    }

    // 将64位输入转换为32位向量 - 对每个小模数取模
    vector<vector<uint32_t>> a_vecs(CRT_CNT);
    vector<vector<uint32_t>> b_vecs(CRT_CNT);
    
    for (int t = 0; t < CRT_CNT; ++t) {
      a_vecs[t].resize(len);
      b_vecs[t].resize(len);
      for (int i = 0; i < len; i++) {
        // 使用巴雷特模乘优化取模运算
        a_vecs[t][i] = static_cast<uint32_t>(barrett_mods[t]->reduce(a[i]));
        b_vecs[t][i] = static_cast<uint32_t>(barrett_mods[t]->reduce(b[i]));
      }
    }

    // 使用OpenMP并行计算NTT
    #pragma omp parallel for
    for (int t = 0; t < CRT_CNT; ++t) {
      ntt_compute_omp(a_vecs[t], b_vecs[t], mods[t], 
                    small_mods[t], root, *barrett_mods[t]);
    }

    // 清零最终输出数组
    fill(ab, ab + 2 * n_ - 1, 0);

    // 创建针对最终大模数的巴雷特计算器
    BarrettReduction barrett_final(p_);
    
    // 使用OpenMP并行进行CRT合并
    #pragma omp parallel
    {
      // 每个线程处理部分结果
      #pragma omp for
      for (int i = 0; i < len; ++i) {
        __uint128_t sum = 0;
        for (int j = 0; j < CRT_CNT; ++j) {
          __uint128_t term = mods[j][i];  // 从32位输入获取
          // 使用巴雷特模乘计算，避免频繁求模
          term = barrett_mods[j]->mul_mod(term, invK[j]);
          term = (term * K[j]) % M;
          sum = (sum + term) % M;
        }
        
        // 只存储有效的结果
        if (i < 2 * n_ - 1) {
          ab[i] = barrett_final.reduce(sum % p_);
        }
      }
    }
    
    auto end = chr::high_resolution_clock::now();
    ans = chr::duration<double, ratio<1, 1000>>(end - start).count();

    fCheck(ab, n_, id);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (ms)" << endl;
    fWrite(ab, n_, id);
  }

  // 释放动态分配的巴雷特模乘计算器
  for (auto &b : barrett_mods) {
    delete b;
  }

  return 0;
}