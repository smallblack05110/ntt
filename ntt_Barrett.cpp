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
using namespace std;

std::string uint128_to_string(__uint128_t value)
{
  if (value == 0)
  {
    return "0";
  }

  // 缓冲区足够存放最大的128位十进制数（39位）和结束符
  char buffer[40];
  int index = 0;

  // 逐位提取数字（反向存储）
  while (value > 0)
  {
    buffer[index++] = '0' + static_cast<char>(value % 10);
    value /= 10;
  }

  // 反转数字顺序得到正确字符串
  std::reverse(buffer, buffer + index);

  // 构造字符串（指定长度避免后续乱码）
  return std::string(buffer, buffer + index);
}

class BarrettMul
{
private:
  uint64_t N;          // 模数

public:
  // 构造函数
  BarrettMul(uint64_t N) : N(N) {
    if (N == 0) {
      throw std::invalid_argument("N must be non-zero");
    }
  }

  // 简化的Barrett模乘：计算 (a * b) mod N
  uint64_t ModMul(uint64_t a, uint64_t b) const {
    // 对输入取模
    a %= N;
    b %= N;

    // 计算乘积
    __uint128_t product = (__uint128_t)a * b;

    // 直接使用128位除法（编译器会优化）
    return (uint64_t)(product % N);
  }

  // 获取模数
  uint64_t getModulus() const {
    return N;
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

__int128_t quick_mod(__int128_t a, __int128_t b, __int128_t p)
{ // 快速计算a的b次方
  __int128_t result = 1;
  a = a % p;
  while (b > 0)
  {
    if (b % 2 == 1)
    {
      result = (result * a) % p; // 奇数就多乘一个a
    }
    a = (a * a) % p; // 底数自乘
    b /= 2;
  }
  return result;
}

void ntt_iter(vector<uint64_t> &a, uint64_t p, int root, bool invert, const BarrettMul &barrett)
{
  int n = a.size();

  // 位反转置换
  for (int i = 1, j = 0; i < n; ++i)
  {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1)
      j ^= bit;
    j |= bit;
    if (i < j)
      swap(a[i], a[j]);
  }

  // NTT主循环
  for (int len = 2; len <= n; len <<= 1)
  {
    uint64_t wn = quick_mod(root, (p - 1) / len, p);
    if (invert)
      wn = quick_mod(wn, p - 2, p);

    for (int i = 0; i < n; i += len)
    {
      uint64_t w = 1;
      for (int j = 0; j < len / 2; ++j)
      {
        uint64_t u = a[i + j];
        uint64_t v = barrett.ModMul(w, a[i + j + len / 2]);
        a[i + j] = (u + v) % p;
        a[i + j + len / 2] = (u - v + p) % p;
        w = barrett.ModMul(w, wn);
      }
    }
  }
}

vector<uint64_t> get_result(vector<uint64_t> &a, vector<uint64_t> &b, int p, int root, const BarrettMul &barrett)
{
  int n = a.size();
  ntt_iter(a, p, root, false, barrett);
  ntt_iter(b, p, root, false, barrett);
  vector<uint64_t> c(n);
  for (int i = 0; i < n; ++i)
    c[i] = barrett.ModMul(a[i], b[i]);
  ntt_iter(c, p, root, true, barrett);
  uint64_t inv_n = quick_mod(n, p - 2, p);
  for (int i = 0; i < n; ++i)
    c[i] = barrett.ModMul(c[i], inv_n);
  return c;
}

__uint128_t power(__uint128_t base, __uint128_t exponent, __uint128_t mod)
{
  __uint128_t result = 1;
  base = base % mod;
  while (exponent > 0)
  {
    if (exponent % 2 == 1)
      result = (result * base) % mod;
    exponent >>= 1;
    base = (base * base) % mod;
  }
  return result;
}

__uint128_t modinv_crt(__uint128_t a, __uint128_t m)
{
  return power(a, m - 2, m);
}

uint64_t a[300000], b[300000], ab[300000];

int main(int argc, char *argv[])
{
  int test_begin = 0, test_end = 4;
  const int root = 3;
  const int CRT_CNT = 4;

  // 查表得到根为3的小模数
  uint64_t small_mods[CRT_CNT] = {
      469762049, 998244353, 1004535809, 1224736769
  };

  // 计算所有模数的乘积
  __uint128_t M = 1;
  for (int i = 0; i < CRT_CNT; i++) {
      M *= small_mods[i];
  }

  // 预计算CRT常量
  __uint128_t K[CRT_CNT];
  __uint128_t invK[CRT_CNT];
  for (int i = 0; i < CRT_CNT; i++) {
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
    while (len < 2 * n_)
      len <<= 1;
    fill(a + n_, a + len, 0);
    fill(b + n_, b + len, 0);

    auto start = chrono::high_resolution_clock::now();

    // 每个小模数下执行NTT
    vector<vector<uint64_t>> mods(CRT_CNT, vector<uint64_t>(len));
    for (int t = 0; t < CRT_CNT; ++t)
    {
        uint64_t m = small_mods[t];
        BarrettMul barrett(m);  // 使用Barrett模乘
        vector<uint64_t> ta(a, a + len), tb(b, b + len);

        auto vc = get_result(ta, tb, m, root, barrett);
        mods[t] = vc;
    }

    // 在CRT合并前清零
    fill(ab, ab + len, 0);

    // CRT合并
    for (int i = 0; i < len; ++i) {
        __uint128_t result = 0;
        for (int j = 0; j < CRT_CNT; ++j) {
            __uint128_t term = mods[j][i];
            term = (term * invK[j]) % small_mods[j];
            term = (term * K[j]) % M;
            result = (result + term) % M;
        }
        ab[i] = result % p_;
    }

    // 还原到原来的模数
    for (int i = 0; i < len; ++i)
    {
      ab[i] = (ab[i] % p_ + p_) % p_;
    }

    auto end = chrono::high_resolution_clock::now();
    ans = chrono::duration<double, ratio<1, 1000>>(end - start).count();

    fCheck(ab, n_, id);
    cout << "average latency for n = " << n_ << " p = " << p_ << " : " << ans << " (us)" << endl;
    fWrite(ab, n_, id);
  }
  return 0;
}
