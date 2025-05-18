#include <cstring>
// #include <string>  // 暂时注释掉，可能有命名冲突
#include <iostream>
#include <fstream>
#include <chrono>  // 保留chrono头文件
#include <iomanip>
#include <sys/time.h>
#include <omp.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <pthread.h> // 引入 pthread 头文件
#include <unistd.h>  // 支持 sysconf 获取 CPU 核心数

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
// 显式引入 chrono 命名空间
namespace chr = std::chrono;  // 使用命名空间别名避免冲突
using std::ratio;  // 从std引入ratio

// Montgomery乘法优化相关结构和常量
struct MontgomeryParams {
    uint32_t mod;     // 原始模数
    uint32_t r;       // R = 2^32
    uint32_t mod_inv; // -mod^(-1) mod R
    uint32_t r2;      // R^2 mod mod
};

// Montgomery乘法 - 计算 (a * b) mod n
uint32_t mont_mul(uint32_t a, uint32_t b, const MontgomeryParams& params) {
    uint64_t t = static_cast<uint64_t>(a) * b;
    uint32_t m = ((uint32_t)t * params.mod_inv) & 0xFFFFFFFF; // 取低32位
    uint64_t res = (t + static_cast<uint64_t>(m) * params.mod) >> 32;
    
    return (res >= params.mod) ? (res - params.mod) : res;
}

// 转换为Montgomery表示 - 计算 (a * R) mod n
uint32_t mont_trans(uint32_t a, const MontgomeryParams& params) {
    return mont_mul(a, params.r2, params);
}

// 从Montgomery表示转回 - 计算 (a / R) mod n
uint32_t mont_reduce(uint32_t a, const MontgomeryParams& params) {
    uint32_t res = mont_mul(a, 1, params);
    return res;
}

// NTT 线程数据结构 - 修改为32位并添加Montgomery参数
struct NTTThreadData
{
  vector<uint32_t> *a;
  vector<uint32_t> *b;
  vector<uint32_t> *result;
  uint64_t p;
  int root;
  MontgomeryParams mont; // 添加Montgomery参数
};

// CRT 重建线程数据结构 - 适应32位 NTT 结果
struct CRTThreadData
{
  vector<vector<uint32_t>> *mods; // 修改为存放各小模 32位NTT 结果的二维向量指针
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

// 计算最大公约数
uint64_t gcd(uint64_t a, uint64_t b) {
    while (b) {
        uint64_t t = b;
        b = a % b;
        a = t;
    }
    return a;
}

// 扩展欧几里得算法求逆元
int64_t extended_gcd(int64_t a, int64_t b, int64_t &x, int64_t &y) {
    if (b == 0) {
        x = 1;
        y = 0;
        return a;
    }
    int64_t x1, y1;
    int64_t d = extended_gcd(b, a % b, x1, y1);
    x = y1;
    y = x1 - y1 * (a / b);
    return d;
}

// 求逆元
int64_t mod_inverse(int64_t a, int64_t m) {
    int64_t x, y;
    extended_gcd(a, m, x, y);
    return (x % m + m) % m;
}

// Montgomery乘法预处理 - 改进版
MontgomeryParams mont_pre(uint64_t mod) {
    MontgomeryParams params;
    params.mod = mod;
    params.r = 0xFFFFFFFF; // R = 2^32 - 1 (用于计算)
    
    // 计算 -mod^(-1) mod 2^32
    int64_t inv = mod_inverse(mod, 0x100000000ULL);
    params.mod_inv = (-inv) & 0xFFFFFFFF;
    
    // 计算 R^2 mod mod
    uint64_t r2 = 1;
    for (int i = 0; i < 64; i++) {
        r2 = (r2 << 1) % mod;
    }
    params.r2 = r2;
    
    return params;
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

// 快速幂
uint64_t quick_pow_mod(uint64_t a, uint64_t b, uint64_t p) {
    uint64_t res = 1;
    a %= p;
    while (b > 0) {
        if (b & 1) res = (res * a) % p;
        a = (a * a) % p;
        b >>= 1;
    }
    return res;
}

// 原始版本的NTT迭代实现
void ntt_iter(vector<uint32_t> &a, uint64_t p, int root, bool invert)
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
    uint64_t wn = quick_pow_mod(root, (p - 1) / len, p);
    if (invert) wn = quick_pow_mod(wn, p - 2, p);
    
    for (int i = 0; i < n; i += len)
    {
      uint32_t w = 1;
      for (int j = 0; j < len / 2; ++j)
      {
        uint32_t u = a[i + j];
        uint64_t v = (static_cast<uint64_t>(a[i + j + len / 2]) * w) % p;
        
        // 使用64位计算，然后再转回32位
        uint32_t sum = (u + v) % p;
        uint64_t diff = (u >= v) ? (u - v) : (u + p - v);
        diff %= p;  // 确保在模数范围内
        
        a[i + j] = sum;
        a[i + j + len/2] = static_cast<uint32_t>(diff);
        
        w = (w * wn) % p;
      }
    }
  }
  
  if (invert) {
    uint64_t inv_n = quick_pow_mod(n, p - 2, p);
    for (int i = 0; i < n; ++i)
      a[i] = static_cast<uint32_t>((static_cast<uint64_t>(a[i]) * inv_n) % p);
  }
}

// NTT 线程入口 - 不使用Montgomery优化
void *ntt_thread_func(void *arg)
{
  auto *d = (NTTThreadData *)arg;
  vector<uint32_t> ta(*d->a), tb(*d->b);
  
  // 使用原始NTT实现
  ntt_iter(ta, d->p, d->root, false);
  ntt_iter(tb, d->p, d->root, false);
  
  // 点乘
  vector<uint32_t> c(ta.size());
  for (size_t i = 0; i < ta.size(); ++i) {
    c[i] = static_cast<uint32_t>((static_cast<uint64_t>(ta[i]) * tb[i]) % d->p);
  }
  
  // 逆变换
  ntt_iter(c, d->p, d->root, true);
  
  *(d->result) = std::move(c);
  return nullptr;
}

vector<uint32_t> get_result_pthread(  //仅调用朴素pthread后的结果
    vector<uint32_t> a,
    vector<uint32_t> b,
    uint64_t p,
    int root)
{
    NTTThreadData data;
    data.a      = &a;
    data.b      = &b;
    data.p      = p;
    data.root   = root;

    vector<uint32_t> res(a.size());
    data.result = &res;

    pthread_t tid;
    pthread_create(&tid, nullptr, ntt_thread_func, &data);
    pthread_join(tid, nullptr);

    return res;
}

// CRT 合并线程入口 - 适应32位输入
void *crt_thread_func(void *arg)
{
  auto *d = (CRTThreadData *)arg;
  for (int i = d->start_idx; i < d->end_idx; ++i)
  {
    __uint128_t sum = 0;
    for (int j = 0; j < d->CRT_CNT; ++j)
    {
      __uint128_t term = (*(d->mods))[j][i];  // 从32位输入获取
      term = (term * d->invK[j]) % d->small_mods[j];
      term = (term * d->K[j])     % d->M;
      sum  = (sum  + term)        % d->M;
    }
    d->ab[i] = uint64_t(sum % d->p_);
  }
  return nullptr;
}

// CRT 模逆
__uint128_t power(__uint128_t base, __uint32_t exp, __uint32_t mod)
{
  __uint128_t res = 1; base %= mod;
  while (exp > 0)
  {
    if (exp & 1) res = (res * base) % mod;
    base = (base * base) % mod; exp >>= 1;
  }
  return res;
}

__uint128_t modinv_crt(__uint128_t a, __uint128_t m)
{
  return power(a, m - 2, m);
}

uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
  // 保证输入的所有模数的原根均为 3, 且模数都能表示为 a \times 4 ^ k + 1 的形式
  // 输入模数分别为 7340033 104857601 469762049 1337006139375617
  // 第四个模数超过了整型表示范围, 如果实现此模数意义下的多项式乘法需要修改框架
  // 对第四个模数的输入数据不做必要要求, 如果要自行探索大模数 NTT, 请在完成前三个模数的基础代码及优化后实现大模数 NTT
  // 输入文件共五个, 第一个输入文件 n = 4, 其余四个文件分别对应四个模数, n = 131072
  // 在实现快速数论变化前, 后四个测试样例运行时间较久, 推荐调试正确性时只使用输入文件 1
  
  // 获取可用 CPU 核心数
  int num_threads = sysconf(_SC_NPROCESSORS_ONLN);
  
  // 设置实际使用线程数（可以根据需要调整）
  int ntt_threads_count = 8; 
  int crt_threads_count = num_threads;  // CRT合并阶段使用所有可用核心
  
  cout << "CPU核心数: " << num_threads << "，使用NTT线程数: " << ntt_threads_count 
       << "，CRT线程数: " << crt_threads_count << endl;

  int test_begin = 0, test_end = 4;
  const int root = 3;
  const int CRT_CNT = 4;  // 小模数数量，也是NTT并行计算的线程数

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
    int len = 1; 
    while (len < 2 * n_) len <<= 1;
    fill(a + n_, a + len, 0);
    fill(b + n_, b + len, 0);

    auto start = chr::high_resolution_clock::now();  // 使用命名空间别名

    // 存储每个小模NTT结果 - 修改为32位
    vector<vector<uint32_t>> mods(CRT_CNT);
    for (int i = 0; i < CRT_CNT; i++) {
        mods[i].resize(len);
    }

    // 创建并启动 NTT 线程
    pthread_t ntt_threads[CRT_CNT];  // 使用固定数量的NTT线程
    NTTThreadData ntt_data[CRT_CNT];

    // 将64位输入转换为32位向量 - 对每个小模数取模
    vector<vector<uint32_t>> a_vecs(CRT_CNT);
    vector<vector<uint32_t>> b_vecs(CRT_CNT);
    
    for (int t = 0; t < CRT_CNT; ++t) {
      a_vecs[t].resize(len);
      b_vecs[t].resize(len);
      for (int i = 0; i < len; i++) {
        a_vecs[t][i] = static_cast<uint32_t>(a[i] % small_mods[t]);
        b_vecs[t][i] = static_cast<uint32_t>(b[i] % small_mods[t]);
      }
    }

    // 启动NTT计算线程
    for (int t = 0; t < CRT_CNT; ++t)
    {
      ntt_data[t].a      = &a_vecs[t];  // 为每个线程使用对应模数的数据
      ntt_data[t].b      = &b_vecs[t];
      ntt_data[t].p      = small_mods[t];  // 使用完整的64位模数
      ntt_data[t].root   = root;
      ntt_data[t].result = &mods[t];
      pthread_create(&ntt_threads[t], nullptr, ntt_thread_func, &ntt_data[t]);
    }

    // 等待所有NTT线程完成
    for (int t = 0; t < CRT_CNT; ++t)
    {
      pthread_join(ntt_threads[t], nullptr);
    }

    // 清零最终输出数组
    fill(ab, ab + 2 * n_ - 1, 0);

    // 创建并启动 CRT 合并线程
    // 限制CRT线程数不超过数据长度
    crt_threads_count = min(crt_threads_count, len);
    pthread_t crt_threads[crt_threads_count];
    CRTThreadData crt_data[crt_threads_count];

    // 计算每个线程处理的数据块大小
    int base_chunk = len / crt_threads_count;
    int rem = len % crt_threads_count;
    int idx = 0;
    
    // 启动CRT合并线程
    for (int t = 0; t < crt_threads_count; ++t)
    {
      int chunk = base_chunk + (t < rem ? 1 : 0);
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
      pthread_create(&crt_threads[t], nullptr, crt_thread_func, &crt_data[t]);
      idx += chunk;
    }

    // 等待所有CRT线程完成
    for (int t = 0; t < crt_threads_count; ++t)
      pthread_join(crt_threads[t], nullptr);

    // 最终还原到大模数
    for (int i = 0; i < 2 * n_ - 1; ++i)
      ab[i] = (ab[i] % p_ + p_) % p_;
    
    auto end = chr::high_resolution_clock::now();  // 使用命名空间别名
    ans = chr::duration<double, ratio<1, 1000>>(end - start).count();  // 使用命名空间别名

    fCheck(ab, n_, id);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (ms)" << endl;
    fWrite(ab, n_, id);
  }

  return 0;
}