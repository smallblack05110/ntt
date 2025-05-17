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
#include <tuple>
#include <pthread.h> // 引入 pthread 头文件
#include <unistd.h>  // 支持 sysconf 获取 CPU 核心数

using namespace std;

// 全局变量：控制线程数
int NTT_THREADS = 4;  // NTT 计算使用的线程数（每个模数一个线程）
int CRT_THREADS = 8;  // CRT 合并使用的线程数

// 提前声明 MontMul 类
class MontMul;

// NTT 线程数据结构
struct NTTThreadData
{
  vector<uint64_t> *a;        // 多项式 a
  vector<uint64_t> *b;        // 多项式 b
  vector<uint64_t> *result;   // 结果存放向量
  uint64_t p;                 // 模数
  int root;                   // 原根
  MontMul *mont;              // Montgomery 乘法对象
};

// CRT 重建线程数据结构
struct CRTThreadData
{
  vector<vector<uint64_t>> *mods; // 存放各小模 NTT 结果的二维向量指针
  uint64_t *ab;                   // 最终结果数组
  int start_idx;                  // 处理起始下标
  int end_idx;                    // 处理结束下标（不含）
  __uint128_t M;                  // 所有小模数乘积
  __uint128_t *K;                 // CRT 常数 K 数组
  __uint128_t *invK;              // CRT 常数 invK 数组
  int64_t p_;                     // 原大模数
  int CRT_CNT;                    // 小模数量
  uint64_t *small_mods;           // 小模数数组
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

class MontMul
{
private:
  uint64_t N;         // 模数
  uint64_t R;         // Montgomery 基（2的幂）
  int logR;           // R 的对数
  uint64_t N_inv_neg; // -N^{-1} mod R
  uint64_t R2;        // R^2 mod N

  struct EgcdResult
  {
    int64_t g;  // 最大公约数
    int64_t x;  // 贝祖系数 x
    int64_t y;  // 贝祖系数 y
  };

  // 扩展欧几里得算法
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

  // 计算模逆元
  static uint64_t modinv(uint64_t a, uint64_t m)
  {
    auto result = egcd(a, m);
    if (result.g != 1)
    {
      throw std::runtime_error("模逆不存在");
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
      throw std::invalid_argument("R 必须是 2 的幂");
    }
    logR = static_cast<int>(std::log2(R));
    if ((1ULL << logR) != R)
    {
      throw std::invalid_argument("R 不是 2 的幂");
    }
    uint64_t N_inv = modinv(N, R);
    N_inv_neg = R - N_inv;
    __int128_t R_squared = static_cast<__int128_t>(R) * R;
    R2 = static_cast<uint64_t>(R_squared % N);
  }

  // REDC 算法，将 __int128_t 类型的 T 转换为 Montgomery 域内元素
  uint64_t REDC(__int128_t T) const
  {
    uint64_t mask = (logR == 64) ? ~0ULL : ((1ULL << logR) - 1);
    uint64_t m_part = static_cast<uint64_t>(T) & mask;
    uint64_t m = (m_part * N_inv_neg) & mask;
    __int128_t mN = static_cast<__int128_t>(m) * N;
    __int128_t t_val = (T + mN) >> logR;
    uint64_t t = static_cast<uint64_t>(t_val);
    return t >= N ? t - N : t;
  }

  // 将普通整数转换到 Montgomery 域
 __int128_t toMont(__int128_t a) const
  {
    return REDC(a * R2);
  }

  // 从 Montgomery 域转换回普通整数
  __int128_t fromMont(__int128_t aR) const
  {
    return REDC(aR);
  }

  // 在 Montgomery 域内进行乘法运算
  __int128_t mulMont(__int128_t aR, __int128_t bR) const
  {
    return REDC(aR * bR);
  }

  // 保持原接口：模乘法
  uint64_t ModMul(uint64_t a, uint64_t b)
  {
    if (a >= N || b >= N)
    {
      throw std::invalid_argument("输入必须小于模数 N");
    }
    uint64_t aR = toMont(a);
    uint64_t bR = toMont(b);
    uint64_t abR = mulMont(aR, bR);
    return fromMont(abR);
  }
};

// 从文件读取输入
void fRead(uint64_t *a, uint64_t *b, int *n, int64_t *p, int input_id)
{
  string str1 = "/nttdata/";
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

// 将结果写入文件
void fWrite(const uint64_t *ab, int n, int input_id)
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

// 校验结果正确性
void fCheck(uint64_t *ab, int n, int input_id){
    // 判断多项式乘法结果是否正确
    std::string str1 = "/nttdata/";
    std::string str2 = std::to_string(input_id);
    std::string strout = str1 + str2 + ".out";
    char data_path[strout.size() + 1];
    std::copy(strout.begin(), strout.end(), data_path);
    data_path[strout.size()] = '\0';
    std::ifstream fin;
    fin.open(data_path, std::ios::in);
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

// 快速幂：计算 a^b mod p
__int128_t quick_mod(__int128_t a, __int128_t b, __int128_t p)
{
  __int128_t res = 1; a %= p;
  while (b > 0)
  {
    if (b & 1) res = (res * a) % p;
    a = (a * a) % p; b >>= 1;
  }
  return res;
}

// 单线程 NTT 迭代实现
void ntt_iter(vector<uint64_t> &a, uint64_t p, int root, bool invert, const MontMul &mont)
{
  int n = a.size();
  
  // 位逆序置换
  for (int i = 1, j = 0; i < n; ++i)
  {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j |= bit;
    if (i < j) swap(a[i], a[j]);
  }
  
  // 蝴蝶操作
  for (int len = 2; len <= n; len <<= 1)
  {
    int wn = quick_mod(root, (p - 1) / len, p);
    if (invert) wn = quick_mod(wn, p - 2, p);
    uint64_t wnR = mont.toMont(wn);
    for (int i = 0; i < n; i += len)
    {
      uint64_t w = mont.toMont(1);
      for (int j = 0; j < len / 2; ++j)
      {
        uint64_t u = a[i + j];
        uint64_t v = mont.mulMont(w, a[i + j + len / 2]);
        a[i + j]         = (u + v) % p;
        a[i + j + len/2] = (u - v + p) % p;
        w = mont.mulMont(w, wnR);
      }
    }
  }
}

// NTT 线程入口函数
void *ntt_thread_func(void *arg)
{
  auto *d = (NTTThreadData *)arg;
  
  // 拷贝数据到本地，避免多线程冲突
  vector<uint64_t> ta(*d->a), tb(*d->b);
  
  // 转 Montgomery 域
  for (size_t i = 0; i < ta.size(); ++i)
  {
    ta[i] = d->mont->toMont(ta[i]);
    tb[i] = d->mont->toMont(tb[i]);
  }
  
  // 正变换 NTT
  ntt_iter(ta, d->p, d->root, false, *d->mont);
  ntt_iter(tb, d->p, d->root, false, *d->mont);
  
  // 点乘
  vector<uint64_t> c(ta.size());
  for (size_t i = 0; i < ta.size(); ++i)
    c[i] = d->mont->mulMont(ta[i], tb[i]);
  
  // 逆变换 INTT
  ntt_iter(c, d->p, d->root, true, *d->mont);
  
  // 乘以 n^{-1}，完成除法
  uint64_t inv_n = d->mont->toMont(quick_mod(ta.size(), d->p - 2, d->p));
  for (size_t i = 0; i < c.size(); ++i)
    c[i] = d->mont->fromMont(d->mont->mulMont(c[i], inv_n));
  
  // 将结果移动到结果向量
  *(d->result) = move(c);
  return nullptr;
}

// 简单包装的获取单个 NTT 结果的函数
vector<uint64_t> get_result_pthread(
    vector<uint64_t> a,
    vector<uint64_t> b,
    uint64_t p,
    int root,
    MontMul &mont)
{
    NTTThreadData data;
    data.a      = &a;
    data.b      = &b;
    data.p      = p;
    data.root   = root;
    data.mont   = &mont;

    vector<uint64_t> res(a.size());
    data.result = &res;

    pthread_t tid;
    pthread_create(&tid, nullptr, ntt_thread_func, &data);
    pthread_join(tid, nullptr);

    return res;
}

// CRT 合并线程入口
void *crt_thread_func(void *arg)
{
  auto *d = (CRTThreadData *)arg;
  
  // 在分配的区间内进行 CRT 合并计算
  for (int i = d->start_idx; i < d->end_idx; ++i)
  {
    __uint128_t sum = 0;
    for (int j = 0; j < d->CRT_CNT; ++j)
    {
      // 计算当前模数系统下的结果
      __uint128_t term = (*d->mods)[j][i];
      term = (term * d->invK[j]) % d->small_mods[j];
      // 乘以系数并取模，防止溢出
      term = (term * d->K[j])     % d->M;
      // 累加到最终结果
      sum  = (sum  + term)        % d->M;
    }
    // 最终模大模数
    d->ab[i] = uint64_t(sum % d->p_);
  }
  return nullptr;
}

// CRT 相关：计算模幂
__uint128_t power(__uint128_t base, __uint128_t exp, __uint128_t mod)
{
  __uint128_t res = 1; base %= mod;
  while (exp > 0)
  {
    if (exp & 1) res = (res * base) % mod;
    base = (base * base) % mod; exp >>= 1;
  }
  return res;
}

// CRT 相关：计算模逆元（费马小定理）
__uint128_t modinv_crt(__uint128_t a, __uint128_t m)
{
  return power(a, m - 2, m);
}

// 全局数组存储多项式和结果
uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
  // 处理命令行参数，控制线程数
  if (argc >= 2) {
    NTT_THREADS = atoi(argv[1]);
  }
  if (argc >= 3) {
    CRT_THREADS = atoi(argv[2]);
  }
  
  // 检测系统可用核心数，如果未指定线程数则使用系统核心数
  int available_cores = sysconf(_SC_NPROCESSORS_ONLN);
  if (NTT_THREADS <= 0) NTT_THREADS = min(4, available_cores);
  if (CRT_THREADS <= 0) CRT_THREADS = min(available_cores, 16);
  
  cout << "使用 NTT 线程数: " << NTT_THREADS << endl;
  cout << "使用 CRT 线程数: " << CRT_THREADS << endl;

  int test_begin = 0, test_end = 4;
  const int root = 3;  // 原根
  const uint64_t R = 1ULL << 31;  // Montgomery 基
  const int CRT_CNT = 4;  // 使用的小模数数量

  // 根为3的小模数列表
  uint64_t small_mods[CRT_CNT] = {
      469762049ULL, 998244353ULL,
      1004535809ULL, 1224736769ULL
  };

  // 计算所有小模数乘积 M
  __uint128_t M = 1;
  for (int i = 0; i < CRT_CNT; ++i) M *= small_mods[i];

  // 预计算 CRT 常量 K 和 invK
  __uint128_t K[CRT_CNT], invK[CRT_CNT];
  for (int i = 0; i < CRT_CNT; ++i)
  {
    K[i] = M / small_mods[i];
    invK[i] = modinv_crt(K[i], small_mods[i]);
  }

  for (int id = test_begin; id <= test_end; ++id)
  {
    long double ans = 0;
    int n_;
    int64_t p_;
    fRead(a, b, &n_, &p_, id);
    
    // 计算最小的2的幂，使其大于等于 2*n_
    int len = 1; 
    while (len < 2 * n_) len <<= 1;
    
    // 填充多项式系数到所需长度
    fill(a + n_, a + len, 0);
    fill(b + n_, b + len, 0);

    auto start = chrono::high_resolution_clock::now();

    // 存储每个小模NTT结果
    vector<vector<uint64_t>> mods(CRT_CNT, vector<uint64_t>(len));

    // 创建并启动 NTT 线程
    int actual_ntt_threads = min(NTT_THREADS, CRT_CNT);
    pthread_t ntt_threads[actual_ntt_threads];
    NTTThreadData ntt_data[actual_ntt_threads];
    MontMul* mont_objs[actual_ntt_threads];

    // 拷贝多项式数据到向量
    vector<uint64_t> a_vec(a, a + len), b_vec(b, b + len);
    
    // 每个线程处理的模数数量
    int mods_per_thread = (CRT_CNT + actual_ntt_threads - 1) / actual_ntt_threads;
    
    // 创建并启动 NTT 线程
    for (int t = 0; t < actual_ntt_threads; ++t)
    {
      // 计算线程处理的模数范围
      int start_mod = t * mods_per_thread;
      int end_mod = min((t + 1) * mods_per_thread, CRT_CNT);
      
      // 如果此线程没有要处理的模数，跳过
      if (start_mod >= CRT_CNT) continue;
      
      // 为每个线程分配的每个模数创建 Montgomery 对象
      for (int mod_idx = start_mod; mod_idx < end_mod; ++mod_idx) {
        mont_objs[mod_idx] = new MontMul(R, small_mods[mod_idx]);
        ntt_data[mod_idx].a      = &a_vec;
        ntt_data[mod_idx].b      = &b_vec;
        ntt_data[mod_idx].p      = small_mods[mod_idx];
        ntt_data[mod_idx].root   = root;
        ntt_data[mod_idx].mont   = mont_objs[mod_idx];
        ntt_data[mod_idx].result = &mods[mod_idx];
        
        // 创建线程
        pthread_create(&ntt_threads[mod_idx], nullptr, ntt_thread_func, &ntt_data[mod_idx]);
      }
    }

    // 等待所有 NTT 线程完成并清理资源
    for (int t = 0; t < CRT_CNT; ++t)
    {
      pthread_join(ntt_threads[t], nullptr);
      delete mont_objs[t];
    }

    // 清零最终输出数组
    fill(ab, ab + 2 * n_ - 1, 0);

    // 创建并启动 CRT 合并线程
    int actual_crt_threads = min(CRT_THREADS, len);
    pthread_t crt_threads[actual_crt_threads];
    CRTThreadData crt_data[actual_crt_threads];

    // 计算每个线程处理的区间
    int base_chunk = len / actual_crt_threads;
    int rem = len % actual_crt_threads;
    int idx = 0;
    
    // 创建并启动 CRT 线程
    for (int t = 0; t < actual_crt_threads; ++t)
    {
      // 计算线程处理的区间大小（处理余数）
      int chunk = base_chunk + (t < rem ? 1 : 0);
      
      // 设置线程参数
      crt_data[t].mods       = &mods;
      crt_data[t].ab         = ab;
      crt_data[t].start_idx  = idx;
      crt_data[t].end_idx    = idx + chunk;
      crt_data[t].M          = M;
      crt_data[t].K          = K;
      crt_data[t].invK       = invK;
      crt_data[t].p_         = p_;
      crt_data[t].CRT_CNT    = CRT_CNT;
      crt_data[t].small_mods = small_mods;
      
      // 创建线程
      pthread_create(&crt_threads[t], nullptr, crt_thread_func, &crt_data[t]);
      idx += chunk;
    }

    // 等待所有 CRT 线程完成
    for (int t = 0; t < actual_crt_threads; ++t)
      pthread_join(crt_threads[t], nullptr);

    // 最终将所有结果还原到大模数
    for (int i = 0; i < 2 * n_ - 1; ++i)
      ab[i] = (ab[i] % p_ + p_) % p_;

    auto end = chrono::high_resolution_clock::now();
    ans = chrono::duration<double, ratio<1, 1000>>(end - start).count();

    // 验证结果和输出
    fCheck(ab, n_, id);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
    fWrite(ab, n_, id);
  }

  return 0;
}