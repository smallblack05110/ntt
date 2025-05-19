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
#include <pthread.h> 
using namespace std;

// MontMul 类的前向声明
class MontMul;

// 并行计算中使用的线程数
const int NUM_THREADS = 8;

// NTT 并行计算的线程参数结构体
struct ThreadParams {
    vector<long long>* a;    // 指向数据向量的指针
    int start;               // 处理区间起始索引
    int end;                 // 处理区间结束索引
    int len;                 // 当前蝶形长度
    int p;                   // 模数 p
    long long wn;            // 原根 wn（未使用，可用于扩展）
    long long wnR;           // Montgomery 域内 wn
    const MontMul* mont;     // 指向 Montgomery 运算对象的指针
    
    // 构造函数用于正确初始化参数
    ThreadParams() : a(nullptr), start(0), end(0), len(0), p(0), wn(0), wnR(0), mont(nullptr) {}
};

class MontMul
{
private:
  uint64_t N;         // 模数 N
  uint64_t R;         // Montgomery 基 R（2 的幂）
  int logR;           // R 的二进制对数
  uint64_t N_inv_neg; // -N^{-1} mod R
  uint64_t R2;        // R^2 mod N

  struct EgcdResult
  {
    int64_t g;
    int64_t x;
    int64_t y;
  };

  // 扩展欧几里得算法，返回 {g=xgcd(a,b), x, y}
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

  // 计算 a 在模 m 下的乘法逆元
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
  // 构造函数，要求 R 为 2 的幂
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
    __int128 R_squared = static_cast<__int128>(R) * R;
    R2 = static_cast<uint64_t>(R_squared % N);
  }

  // REDC 算法：将 __int128 类型的 T 转换到 Montgomery 域内
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

  // 保留原有接口：对于 a, b（均小于 N），返回 a * b mod N
  uint64_t ModMul(uint64_t a, uint64_t b)
  {
    if (a >= N || b >= N)
    {
      throw std::invalid_argument("输入必须小于模 N");
    }
    uint64_t aR = toMont(a);
    uint64_t bR = toMont(b);
    uint64_t abR = mulMont(aR, bR);
    return fromMont(abR);
  }
};

// 从文件读取输入数据：多项式长度 n，模数 p，和多项式系数 a, b
void fRead(int *a, int *b, int *n, int *p, int input_id)
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

// 将结果写入输出文件
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

// 校验输出是否正确
void fCheck(int *ab, int n, int input_id)
{
  string str1 = "/nttdata/";
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

// 快速幂：计算 a^b mod p
int quick_mod(int a, int b, int p)
{
  int result = 1;
  a = a % p;
  while (b > 0)
  {
    if (b % 2 == 1)
    {
      result = (1LL * result * a) % p; // 若是奇数，乘一次 a
    }
    a = (1LL * a * a) % p; // 平方
    b /= 2;
  }
  return result;
}

// 线程执行的 NTT 计算函数
void* ntt_thread_func(void* arg) {
    ThreadParams* params = (ThreadParams*)arg;
    vector<long long>* a = params->a;
    int start = params->start;
    int end = params->end;
    int len = params->len;
    int p = params->p;
    long long wnR = params->wnR;
    const MontMul* mont = params->mont;

    // 蝶形计算区间并行
    for (int i = start; i < end; i += len) {
        long long w = mont->toMont(1);
        for (int j = 0; j < len / 2; ++j) {
            long long u = (*a)[i + j];
            long long v = mont->mulMont(w, (*a)[i + j + len / 2]);
            (*a)[i + j] = (u + v) % p;
            (*a)[i + j + len / 2] = (u - v + p) % p;
            w = mont->mulMont(w, wnR);
        }
    }
    
    return NULL;
}

// 点乘阶段的线程函数
void* pointwise_multiply_thread(void* arg) {
    ThreadParams* params = (ThreadParams*)arg;
    vector<long long>* a = params->a;
    const MontMul* mont = params->mont;
    int start = params->start;
    int end = params->end;
    int p = params->p;
    vector<long long>* b = (vector<long long>*)(params->wn);  // 重用 wn 保存 b 的地址
    vector<long long>* c = (vector<long long>*)(params->wnR); // 重用 wnR 保存 c 的地址
    
    for (int i = start; i < end; ++i) {
        (*c)[i] = mont->mulMont((*a)[i], (*b)[i]) % p;
    }
    
    return NULL;
}

// 使用 pthread 并行的 NTT 迭代版
void ntt_iter_pthread(vector<long long> &a, int p, int root, bool invert, const MontMul &mont)
{
    int n = a.size();
    
    // 位逆序调整
    for (int i = 1, j = 0; i < n; ++i)
    {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1)
            j ^= bit;
        j |= bit;
        if (i < j)
            swap(a[i], a[j]);
    }

    pthread_t threads[NUM_THREADS];
    ThreadParams params[NUM_THREADS];

    // 按长度迭代进行蝶形计算
    for (int len = 2; len <= n; len <<= 1)
    {
        int wn = quick_mod(root, (p - 1) / len, p);
        if (invert)
            wn = quick_mod(wn, p - 2, p);
        long long wnR = mont.toMont(wn);

        int chunk_size = n / NUM_THREADS;
        if (chunk_size < len) {
            // 区间小于蝶形长度时顺序执行
            for (int i = 0; i < n; i += len)
            {
                long long w = mont.toMont(1);
                for (int j = 0; j < len / 2; ++j)
                {
                    long long u = a[i + j];
                    long long v = mont.mulMont(w, a[i + j + len / 2]);
                    a[i + j] = (u + v) % p;
                    a[i + j + len / 2] = (u - v + p) % p;
                    w = mont.mulMont(w, wnR);
                }
            }
        } else {
            // 否则并行处理
            for (int t = 0; t < NUM_THREADS; ++t) {
                int start = t * chunk_size;
                int end = (t == NUM_THREADS - 1) ? n : (t + 1) * chunk_size;
                start = (start / len) * len; // 对齐到 len 的倍数
                
                params[t].a = &a;
                params[t].start = start;
                params[t].end = end;
                params[t].len = len;
                params[t].p = p;
                params[t].wnR = wnR;
                params[t].mont = &mont;
                pthread_create(&threads[t], NULL, ntt_thread_func, &params[t]);
            }
            
            // 等待所有线程完成
            for (int t = 0; t < NUM_THREADS; ++t) {
                pthread_join(threads[t], NULL);
            }
        }
    }
}

// 最原始的朴素多项式乘法
void poly_multiply(int *a, int *b, int *ab, int n, int p){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            ab[i+j]=(1LL * a[i] * b[j] % p + ab[i+j]) % p;
        }
    }
}

// 使用 pthread 并行的 get_result，实现完整的 NTT 多项式乘法流程
vector<long long> get_result_pthread(vector<long long> &a, vector<long long> &b, int p, int root, const MontMul &mont)
{
    int n = a.size();
    
    // 正向 NTT
    ntt_iter_pthread(a, p, root, false, mont);
    ntt_iter_pthread(b, p, root, false, mont);
    
    // 并行点乘阶段
    vector<long long> c(n);
    pthread_t threads[NUM_THREADS];
    ThreadParams params[NUM_THREADS];
    
    int chunk_size = n / NUM_THREADS;
    for (int t = 0; t < NUM_THREADS; ++t) {
        int start = t * chunk_size;
        int end = (t == NUM_THREADS - 1) ? n : (t + 1) * chunk_size;
        
        // 重用 wn 和 wnR 字段传递 b, c 向量地址
        params[t].a = &a;
        params[t].start = start;
        params[t].end = end;
        params[t].p = p;
        params[t].wn = (long long)&b;
        params[t].wnR = (long long)&c;
        params[t].mont = &mont;
        pthread_create(&threads[t], NULL, pointwise_multiply_thread, &params[t]);
    }
    
    // 等待所有线程完成
    for (int t = 0; t < NUM_THREADS; ++t) {
        pthread_join(threads[t], NULL);
    }
    
    // 逆向 NTT
    ntt_iter_pthread(c, p, root, true, mont);
    
    // 缩放结果
    int inv_n = quick_mod(n, p - 2, p);
    long long invR = mont.toMont(inv_n);
    
    // 并行缩放
    for (int t = 0; t < NUM_THREADS; ++t) {
        int start = t * chunk_size;
        int end = (t == NUM_THREADS - 1) ? n : (t + 1) * chunk_size;
        
        for (int i = start; i < end; ++i) {
            c[i] = mont.mulMont(c[i], invR);
        }
    }
    
    return c;
}

int a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
    // 确保输入模数的原根均为 3，且模数可表示为 a * 4^k + 1
    // 输入模数分别为 7340033, 104857601, 469762049, 263882790666241
    // 第四个模数超过整型范围，如需支持大模数 NTT，请在前三个基础上自行扩展
    // 输入文件共五个，第一个样例 n=4，其余四个样例 n=131072
    // 在调试正确性期间，建议仅使用第一个样例以节约时间
    int test_begin = 0, test_end = 4;
    for (int id = test_begin; id <= test_end; ++id)
    {
        long double ans = 0;
        int n_, p_;
        fRead(a, b, &n_, &p_, id);

        // 初始化输出数组为 0
        memset(ab, 0, sizeof(int) * (2 * n_));

        int len = 1;
        while (len < 2 * n_)
            len <<= 1;
        fill(a + n_, a + len, 0);
        fill(b + n_, b + len, 0);

        vector<long long> va(a, a + len), vb(b, b + len);
        long long R = 1LL << 30;
        MontMul mont(R, p_);
        for (int i = 0; i < len; ++i)
        {
            va[i] = mont.toMont(va[i]);
            vb[i] = mont.toMont(vb[i]);
        }

        int root = 3;
        auto start = chrono::high_resolution_clock::now();
        
        // 使用 pthread 版本的 NTT 多项式乘法
        vector<long long> cr = get_result_pthread(va, vb, p_, root, mont);
        
        auto end = chrono::high_resolution_clock::now();
        ans = chrono::duration<double, ratio<1, 1000>>(end - start).count();

        for (int i = 0; i < 2 * n_ - 1; ++i)
        {
            ab[i] = (int)mont.fromMont(cr[i]);
        }

        fCheck(ab, n_, id);
        cout << "n = " << n_ << " p = " << p_ << " (pthread) 平均耗时(ms): " << ans << endl;
        fWrite(ab, n_, id);
    }
    return 0;
}
