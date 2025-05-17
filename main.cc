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

// 提前声明 MontMul 类
class MontMul;

// NTT 线程数据结构
struct NTTThreadData
{
  vector<uint64_t> *a;
  vector<uint64_t> *b;
  vector<uint64_t> *result;
  uint64_t p;
  int root;
  MontMul *mont;
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

// 快速幂
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

// 单线程 NTT
void ntt_iter(vector<uint64_t> &a, uint64_t p, int root, bool invert, const MontMul &mont)
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

// NTT 线程入口
void *ntt_thread_func(void *arg)
{
  auto *d = (NTTThreadData *)arg;
  vector<uint64_t> ta(*d->a), tb(*d->b);
  // 转 Montgomery 域
  for (size_t i = 0; i < ta.size(); ++i)
  {
    ta[i] = d->mont->toMont(ta[i]);
    tb[i] = d->mont->toMont(tb[i]);
  }
  // 正、逆 NTT
  ntt_iter(ta, d->p, d->root, false, *d->mont);
  ntt_iter(tb, d->p, d->root, false, *d->mont);
  // 点乘
  vector<uint64_t> c(ta.size());
  for (size_t i = 0; i < ta.size(); ++i)
    c[i] = d->mont->mulMont(ta[i], tb[i]);
  // 逆变换
  ntt_iter(c, d->p, d->root, true, *d->mont);
  // 乘以 n^{-1}
  uint64_t inv_n = d->mont->toMont(quick_mod(ta.size(), d->p - 2, d->p));
  for (size_t i = 0; i < c.size(); ++i)
    c[i] = d->mont->fromMont(d->mont->mulMont(c[i], inv_n));
  *(d->result) = move(c);
  return nullptr;
}

vector<uint64_t> get_result_pthread(  //仅调用朴素pthread后的结果
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
  for (int i = d->start_idx; i < d->end_idx; ++i)
  {
    __uint128_t sum = 0;
    for (int j = 0; j < d->CRT_CNT; ++j)
    {
      __uint128_t term = (*d->mods)[j][i];
      term = (term * d->invK[j]) % d->small_mods[j];
      term = (term * d->K[j])     % d->M;
      sum  = (sum  + term)        % d->M;
    }
    d->ab[i] = uint64_t(sum % d->p_);
  }
  return nullptr;
}

// CRT 模逆
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
  cout << "使用线程数: " << num_threads << endl;

  int test_begin = 0, test_end = 4;
  const int root = 3;
  const uint64_t R = 1ULL << 31;
  const int CRT_CNT = 4;

  // 根为3的小模数列表
  uint64_t small_mods[CRT_CNT] = {
      469762049ULL, 998244353ULL,
      1004535809ULL,1224736769ULL
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

    auto start = chrono::high_resolution_clock::now();

    // 存储每个小模NTT结果
    vector<vector<uint64_t>> mods(CRT_CNT, vector<uint64_t>(len));

    // 创建并启动 NTT 线程
    pthread_t ntt_threads[CRT_CNT];
    NTTThreadData ntt_data[CRT_CNT];
    MontMul* mont_objs[CRT_CNT];

    vector<uint64_t> a_vec(a, a + len), b_vec(b, b + len);
    for (int t = 0; t < CRT_CNT; ++t)
    {
      mont_objs[t] = new MontMul(R, small_mods[t]);
      ntt_data[t].a      = &a_vec;
      ntt_data[t].b      = &b_vec;
      ntt_data[t].p      = small_mods[t];
      ntt_data[t].root   = root;
      ntt_data[t].mont   = mont_objs[t];
      ntt_data[t].result = &mods[t];
      pthread_create(&ntt_threads[t], nullptr, ntt_thread_func, &ntt_data[t]);
    }

    // 等待 NTT 线程完成并清理
    for (int t = 0; t < CRT_CNT; ++t)
    {
      pthread_join(ntt_threads[t], nullptr);
      delete mont_objs[t];
    }

    // 清零最终输出数组
    fill(ab, ab + 2 * n_ - 1, 0);

    // 创建并启动 CRT 合并线程
    int crt_threads_count = min(num_threads, len);
    pthread_t crt_threads[crt_threads_count];
    CRTThreadData crt_data[crt_threads_count];

    int base_chunk = len / crt_threads_count;
    int rem = len % crt_threads_count;
    int idx = 0;
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

    // 等待 CRT 线程完成
    for (int t = 0; t < crt_threads_count; ++t)
      pthread_join(crt_threads[t], nullptr);

    // 最终还原到大模数
    for (int i = 0; i < 2 * n_ - 1; ++i)
      ab[i] = (ab[i] % p_ + p_) % p_;
    // vector<uint64_t> va(a, a+len), vb(b, b+len);
    // MontMul mont(R, p_);
    // auto vc = get_result_pthread(va, vb, p_, root, mont);
    // for (int i = 0; i < 2*n_ - 1; ++i) {
    //     ab[i] = vc[i];
    // }
     auto end = chrono::high_resolution_clock::now();
    ans = chrono::duration<double, ratio<1, 1000>>(end - start).count();

    fCheck(ab, n_, id);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
    fWrite(ab, n_, id);
  }

  return 0;
}
