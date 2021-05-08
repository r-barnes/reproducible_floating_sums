//Compile with:
//g++ -g -O3 repro_vector.cpp -fopenmp

//NOTE: Always comile with `-g`. It doesn't slow down your code, but does make
//it debuggable and improves ease of profiling

#include <algorithm>
#include <bitset>               //Used for showing bitwise representations
#include <cfenv>                //Used for setting floating-point rounding modes
#include <chrono>               //Used for timing algorithms
#include <climits>              //Used for showing bitwise representations
#include <iostream>
#include <omp.h>                //OpenMP
#include <random>
#include <stdexcept>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>



// Simple timer class for tracking cumulative run time of the different
// algorithms
struct Timer {
  double total = 0;
  std::chrono::high_resolution_clock::time_point start_time;

  Timer() = default;

  void start() {
    start_time = std::chrono::high_resolution_clock::now();
  }

  void stop() {
    const auto now = std::chrono::high_resolution_clock::now();
    const auto time_span = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time);
    total += time_span.count();
  }
};

//Simple class to enable directed rounding in floating-point math and to reset
//the rounding mode afterwards, when it goes out of scope
struct SetRoundingMode {
  const int old_rounding_mode;

  SetRoundingMode(const int mode) : old_rounding_mode(fegetround()) {
    if(std::fesetround(mode)!=0){
      throw std::runtime_error("Failed to set directed rounding mode!");
    }
  }

  ~SetRoundingMode(){
    if(std::fesetround(old_rounding_mode)!=0){
      throw std::runtime_error("Failed to reset rounding mode to original value!");
    }
  }

  static std::string get_rounding_mode_string() {
    switch (fegetround()) {
      case FE_DOWNWARD:   return "downward";
      case FE_TONEAREST:  return "to-nearest";
      case FE_TOWARDZERO: return "toward-zero";
      case FE_UPWARD:     return "upward";
      default:            return "unknown";
    }
  }
};

// Used to make showing bitwise representations somewhat more intuitive
template<class T>
struct binrep {
  const T val;
  binrep(const T val0) : val(val0) {}
};

// Display the bitwise representation
template<class T>
std::ostream& operator<<(std::ostream& out, const binrep<T> a){
  const char* beg = reinterpret_cast<const char*>(&a.val);
  const char *const end = beg + sizeof(a.val);
  while(beg != end){
    out << std::bitset<CHAR_BIT>(*beg++);
    if(beg < end)
      out << ' ';
  }
  return out;
}



//Simple serial summation algorithm with an accumulation type we can specify
//to more fully explore its behaviour
template<class FloatType, class SimpleAccumType>
FloatType serial_simple_summation(const std::vector<FloatType> &vec){
  SimpleAccumType sum = 0;
  for(const auto &x: vec){
    sum += x;
  }
  return sum;
}

//Parallel variant of the simple summation algorithm above
template<class FloatType, class SimpleAccumType>
FloatType parallel_simple_summation(const std::vector<FloatType> &vec){
  SimpleAccumType sum = 0;
  #pragma omp parallel for default(none) reduction(+:sum) shared(vec)
  for(size_t i=0;i<vec.size();i++){
    sum += vec[i];
  }
  return sum;
}



//Kahan's compensated summation algorithm for accurately calculating sums of
//many numbers with O(1) error
template<class FloatType>
FloatType serial_kahan_summation(const std::vector<FloatType> &vec){
  FloatType sum = 0.0f;
  FloatType c = 0.0f;
  for (const auto &num: vec) {
    const auto y = num - c;
    const auto t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
}

//Parallel version of Kahan's compensated summation algorithm (could be improved
//by better accounting for the compsenation during the reduction phase)
template<class FloatType>
FloatType parallel_kahan_summation(const std::vector<FloatType> &vec){
  //Parallel phase
  std::vector<FloatType> sum(omp_get_max_threads(), 0);
  FloatType c = 0;
  #pragma omp parallel for default(none) firstprivate(c) shared(sum,vec)
  for (size_t i=0;i<vec.size();i++) {
    const auto tid = omp_get_thread_num();
    const auto y = vec[i] - c;
    const auto t = sum.at(tid) + y;
    c = (t - sum[tid]) - y;
    sum[tid] = t;
  }

  //Serial reduction phase

  //This could be more accurate if it took the remaining compensation values
  //from above into account
  FloatType total_sum = 0.0f;
  FloatType total_c = 0.0f;
  for(const auto &num: sum){
    const auto y = num - total_c;
    const auto t = total_sum + y;
    total_c = (t - total_sum) - y;
    total_sum = t;
  }

  return total_sum;
}



// Error-free vector transformation. Algorithm 4 from Demmel and Nguyen (2013)
template<class FloatType>
FloatType ExtractVectorNew2(
  const double M,
  const typename std::vector<FloatType>::iterator &begin,
  const typename std::vector<FloatType>::iterator &end
){
  // Should use the directed rounding mode of the parent thread

  auto Mold = M;
  for(auto v=begin;v!=end;v++){
    auto Mnew = Mold + (*v);
    auto q = Mnew - Mold;
    (*v) -= q;
    Mold = Mnew;
  }

  //This is the exact sum of high order parts q_i
  //v is now the vector of low order parts r_i
  return Mold - M;
}

//Serial bitwise deterministic summation.
//Algorithm 8 from Demmel and Nguyen (2013).
template<class FloatType>
FloatType serial_bitwise_deterministic_summation(
  std::vector<FloatType> vec,  // Note that we're making a copy!
  const int k
){
  constexpr FloatType eps = std::numeric_limits<FloatType>::epsilon();
  const auto n = vec.size();
  const auto adr = SetRoundingMode(FE_UPWARD);

  if(n==0){
    return 0;
  }

  FloatType m = std::abs(vec.front());
  for(const auto &x: vec){
    m = std::max(m, std::abs(x));
  }

  FloatType delta_f = n * m / (1 - 4 * (n + 1) * eps);
  FloatType Mf = 3 * std::pow(2,std::ceil(std::log2(delta_f)));

  std::vector<FloatType> Tf(k);
  for(int f=0;f<k-1;f++){
    Tf[f] = ExtractVectorNew2<FloatType>(Mf, vec.begin(), vec.end());
    delta_f = n * (4 * eps * Mf / 3) / (1 - 4 * (n + 1) * eps);
    Mf = 3 * std::pow(2,std::ceil(std::log2(delta_f)));
  }

  FloatType M = Mf;
  for(const FloatType &v: vec){
    M += v;
  }
  Tf[k-1] = M - Mf;

  FloatType T = 0;
  for(const FloatType &tf: Tf){
    T += tf;
  }

  return T;
}

//Parallel bitwise deterministic summation.
//Algorithm 9 from Demmel and Nguyen (2013).
template<class FloatType>
FloatType parallel_bitwise_deterministic_summation(
  std::vector<FloatType> vec,  // Note that we're making a copy!
  const int k
){
  constexpr FloatType eps = std::numeric_limits<FloatType>::epsilon();
  const auto n = vec.size();
  const auto adr = SetRoundingMode(FE_UPWARD);

  if(n==0){
    return 0;
  }

  std::vector<FloatType> Tf(k);

  FloatType m = std::abs(vec.front());
  #pragma omp parallel for default(none) reduction(max:m) shared(vec)
  for(size_t i=0;i<vec.size();i++){
    m = std::max(m, std::abs(vec[i]));
  }

  #pragma omp declare reduction(vec_plus : std::vector<FloatType> : \
    std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<FloatType>())) \
    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

  #pragma omp parallel default(none) reduction(vec_plus:Tf) shared(k,m,n,vec)
  {
    const auto adr = SetRoundingMode(FE_UPWARD);
    const auto threads = omp_get_num_threads();
    const auto tid = omp_get_thread_num();
    const auto values_per_thread = n / threads;
    const auto nlow = tid * values_per_thread;
    const auto nhigh = (tid<threads-1) ? ((tid+1) * values_per_thread) : n;

    FloatType delta_f = n * m / (1 - 4 * (n + 1) * eps);
    FloatType Mf = 3 * std::pow(2,std::ceil(std::log2(delta_f)));

    for(int f=0;f<k-1;f++){
      Tf[f] = ExtractVectorNew2<FloatType>(Mf, vec.begin() + nlow, vec.begin() + nhigh);
      delta_f = n * (4 * eps * Mf / 3) / (1 - 4 * (n + 1) * eps);
      Mf = 3 * std::pow(2,std::ceil(std::log2(delta_f)));
    }

    FloatType M = Mf;
    for(size_t i=nlow;i<nhigh;i++){
      M += vec[i];
    }
    Tf[k-1] = M - Mf;
  }

  FloatType T = 0;
  for(const FloatType &tf: Tf){
    T += tf;
  }

  return T;
}



//Convenience wrappers
template<bool Parallel, class FloatType>
FloatType bitwise_deterministic_summation(
  const std::vector<FloatType> &vec,  // Note that we're making a copy!
  const int k
){
  if(Parallel){
    return parallel_bitwise_deterministic_summation<FloatType>(vec, k);
  } else {
    return serial_bitwise_deterministic_summation<FloatType>(vec, k);
  }
}

template<bool Parallel, class FloatType, class SimpleAccumType>
FloatType simple_summation(const std::vector<FloatType> &vec){
  if(Parallel){
    return parallel_simple_summation<FloatType, SimpleAccumType>(vec);
  } else {
    return serial_simple_summation<FloatType, SimpleAccumType>(vec);
  }
}

template<bool Parallel, class FloatType>
FloatType kahan_summation(const std::vector<FloatType> &vec){
  if(Parallel){
    return serial_kahan_summation<FloatType>(vec);
  } else {
    return parallel_kahan_summation<FloatType>(vec);
  }
}



// Timing tests for the summation algorithms
template<bool Parallel, class FloatType, class SimpleAccumType>
FloatType PerformTestsOnData(
  const int TESTS,
  std::vector<FloatType> floats, //Make a copy so we use the same data for each test
  std::mt19937 gen               //Make a copy so we use the same data for each test
){
  Timer time_deterministic;
  Timer time_kahan;
  Timer time_simple;

  //Very precise output
  std::cout.precision(std::numeric_limits<FloatType>::max_digits10);
  std::cout<<std::fixed;

  //Get a reference value
  std::unordered_map<FloatType, uint32_t> simple_sums;
  std::unordered_map<FloatType, uint32_t> kahan_sums;
  const auto ref_val = bitwise_deterministic_summation<Parallel, FloatType>(floats, 2);
  for(int test=0;test<TESTS;test++){
    std::shuffle(floats.begin(), floats.end(), gen);

    time_deterministic.start();
    const auto my_val = bitwise_deterministic_summation<Parallel, FloatType>(floats, 2);
    time_deterministic.stop();
    if(ref_val!=my_val){
      std::cout<<"ERROR: UNEQUAL VALUES ON TEST #"<<test<<"!"<<std::endl;
      std::cout<<"Reference      = "<<ref_val                   <<std::endl;
      std::cout<<"Current        = "<<my_val                    <<std::endl;
      std::cout<<"Reference bits = "<<binrep<FloatType>(ref_val)<<std::endl;
      std::cout<<"Current   bits = "<<binrep<FloatType>(my_val) <<std::endl;
      throw std::runtime_error("Values were not equal!");
    }

    time_kahan.start();
    const auto kahan_sum = kahan_summation<Parallel, FloatType>(floats);
    kahan_sums[kahan_sum]++;
    time_kahan.stop();

    time_simple.start();
    const auto simple_sum = simple_summation<Parallel, FloatType, SimpleAccumType>(floats);
    simple_sums[simple_sum]++;
    time_simple.stop();
  }

  std::cout<<"Parallel? "<<Parallel<<std::endl;
  if(Parallel){
    std::cout<<"Max threads = "<<omp_get_max_threads()<<std::endl;
  }
  std::cout<<"Floating type                        = "<<typeid(FloatType).name()<<std::endl;
  std::cout<<"Simple summation accumulation type   = "<<typeid(SimpleAccumType).name()<<std::endl;
  std::cout<<"Average deterministic summation time = "<<(time_deterministic.total/TESTS)<<std::endl;
  std::cout<<"Average simple summation time        = "<<(time_simple.total/TESTS)<<std::endl;
  std::cout<<"Average Kahan summation time         = "<<(time_kahan.total/TESTS)<<std::endl;
  std::cout<<"Ratio Deterministic to Simple        = "<<(time_deterministic.total/time_simple.total)<<std::endl;
  std::cout<<"Ratio Deterministic to Kahan         = "<<(time_deterministic.total/time_kahan.total)<<std::endl;

  std::cout<<"Reference value                      = "<<std::fixed<<ref_val<<std::endl;
  std::cout<<"Reference bits                       = "<<binrep<FloatType>(ref_val)<<std::endl;

  std::cout<<"Distinct Kahan values                = "<<kahan_sums.size()<<std::endl;
  std::cout<<"Distinct Simple values               = "<<simple_sums.size()<<std::endl;

  for(const auto &kv: kahan_sums){
    // std::cout<<"Kahan sum values (N="<<std::fixed<<kv.second<<") "<<kv.first<<" ("<<binrep<FloatType>(kv.first)<<")"<<std::endl;
  }

  for(const auto &kv: simple_sums){
    // std::cout<<"Simple sum values (N="<<std::fixed<<kv.second<<") "<<kv.first<<" ("<<binrep<FloatType>(kv.first)<<")"<<std::endl;
  }

  std::cout<<std::endl;

  return ref_val;
}

// Use this to make sure the tests are reproducible
template<class FloatType, class SimpleAccumType>
void PerformTests(const int N, const int TESTS){
  std::random_device rd;
  // std::mt19937 gen(rd());
  std::mt19937 gen(123456789);
  std::uniform_real_distribution<FloatType> distr(-10000, 10000);
  std::vector<FloatType> floats;
  for(int i=0;i<N;i++){
    floats.push_back(distr(gen));
  }
  const auto serial_val = PerformTestsOnData<false, FloatType, SimpleAccumType>(TESTS, floats, gen);
  const auto parallel_val = PerformTestsOnData<true, FloatType, SimpleAccumType>(TESTS, floats, gen);

  //Note that the `long double` type may only use 12-16 bytes (to maintain
  //alignment), but only 80 bits, resulting in bitwise indeterminism in the last
  //few bits; however, the floating-point values themselves will be equal.
  std::cout<<"Serial and Parallel values match for "
           <<typeid(FloatType).name()
           <<"? "
           <<(serial_val==parallel_val)
           <<"\n"<<std::endl;
}



int main(){
  const int N = 1'000'000;
  const int TESTS = 100;

  // PerformTests<float, float>(N, TESTS);
  // PerformTests<float, double>(N, TESTS);
  PerformTests<double, double>(N, TESTS);
  PerformTests<long double, long double>(N, TESTS);

  return 0;
}
