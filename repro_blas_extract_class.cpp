//g++ -O3 -g --std=c++17 repro_blas_extract_class.cpp -Wall
#include "reproducible_floating_accumulator.hpp"

#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <unordered_map>
#include <vector>

//Kahan's compensated summation algorithm for accurately calculating sums of
//many numbers with O(1) error
template<class AccumType, class FloatType>
FloatType serial_kahan_summation(const std::vector<FloatType> &vec){
  AccumType sum = 0.0f;
  AccumType c = 0.0f;
  for (const auto &num: vec) {
    const auto y = num - c;
    const auto t = sum + y;
    c = (t - sum) - y;
    sum = t;
  }
  return sum;
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



template<class FloatType>
FloatType bitwise_deterministic_summation_1(const std::vector<FloatType> &vec){
  ReproducibleFloatingAccumulator<FloatType> rfa;
  for(const auto &x: vec){
    rfa += x;
  }
  return rfa.conv();
}

template<class FloatType>
FloatType bitwise_deterministic_summation_many(const std::vector<FloatType> &vec){
  ReproducibleFloatingAccumulator<FloatType> rfa;
  rfa.add(vec.begin(), vec.end());
  return rfa.conv();
}

template<class FloatType>
FloatType bitwise_deterministic_summation_manyc(const std::vector<FloatType> &vec, const FloatType max_abs_val){
  ReproducibleFloatingAccumulator<FloatType> rfa;
  rfa.add(vec.begin(), vec.end(), max_abs_val);
  return rfa.conv();
}


// Timing tests for the summation algorithms
template<class FloatType, class SimpleAccumType>
FloatType PerformTestsOnData(
  const int TESTS,
  std::vector<FloatType> floats, //Make a copy so we use the same data for each test
  std::mt19937 gen               //Make a copy so we use the same data for each test
){
  Timer time_deterministic_1;
  Timer time_deterministic_many;
  Timer time_deterministic_manyc;
  Timer time_kahan;
  Timer time_simple;

  //Very precise output
  std::cout.precision(std::numeric_limits<FloatType>::max_digits10);
  std::cout<<std::fixed;

  std::cout<<"Floating type                        = "<<typeid(FloatType).name()<<std::endl;
  std::cout<<"Simple summation accumulation type   = "<<typeid(SimpleAccumType).name()<<std::endl;

  //Get a reference value
  std::unordered_map<FloatType, uint32_t> simple_sums;
  std::unordered_map<FloatType, uint32_t> kahan_sums;
  const auto ref_val = bitwise_deterministic_summation_1<FloatType>(floats);
  for(int test=0;test<TESTS;test++){
    std::shuffle(floats.begin(), floats.end(), gen);

    time_deterministic_1.start();
    const auto my_val_1 = bitwise_deterministic_summation_1<FloatType>(floats);
    time_deterministic_1.stop();
    if(ref_val!=my_val_1){
      std::cout<<"ERROR: UNEQUAL VALUES ON TEST #"<<test<<" for add-1!"<<std::endl;
      std::cout<<"Reference      = "<<ref_val                    <<std::endl;
      std::cout<<"Current        = "<<my_val_1                   <<std::endl;
      std::cout<<"Reference bits = "<<binrep<FloatType>(ref_val) <<std::endl;
      std::cout<<"Current   bits = "<<binrep<FloatType>(my_val_1)<<std::endl;
      throw std::runtime_error("Values were not equal!");
    }

    time_deterministic_many.start();
    const auto my_val_many = bitwise_deterministic_summation_many<FloatType>(floats);
    time_deterministic_many.stop();
    if(ref_val!=my_val_many){
      std::cout<<"ERROR: UNEQUAL VALUES ON TEST #"<<test<<" for add-many!"<<std::endl;
      std::cout<<"Reference      = "<<ref_val                       <<std::endl;
      std::cout<<"Current        = "<<my_val_many                   <<std::endl;
      std::cout<<"Reference bits = "<<binrep<FloatType>(ref_val)    <<std::endl;
      std::cout<<"Current   bits = "<<binrep<FloatType>(my_val_many)<<std::endl;
      throw std::runtime_error("Values were not equal!");
    }

    time_deterministic_manyc.start();
    const auto my_val_manyc = bitwise_deterministic_summation_manyc<FloatType>(floats, 1000);
    time_deterministic_manyc.stop();
    if(ref_val!=my_val_manyc){
      std::cout<<"ERROR: UNEQUAL VALUES ON TEST #"<<test<<" for add-many!"<<std::endl;
      std::cout<<"Reference      = "<<ref_val                        <<std::endl;
      std::cout<<"Current        = "<<my_val_manyc                   <<std::endl;
      std::cout<<"Reference bits = "<<binrep<FloatType>(ref_val)     <<std::endl;
      std::cout<<"Current   bits = "<<binrep<FloatType>(my_val_manyc)<<std::endl;
      throw std::runtime_error("Values were not equal!");
    }

    time_kahan.start();
    const auto kahan_sum = serial_kahan_summation<FloatType>(floats);
    kahan_sums[kahan_sum]++;
    time_kahan.stop();

    time_simple.start();
    const auto simple_sum = serial_simple_summation<FloatType, SimpleAccumType>(floats);
    simple_sums[simple_sum]++;
    time_simple.stop();
  }

  std::cout<<"Average deterministic sum 1ata time  = "<<(time_deterministic_1.total/TESTS)<<std::endl;
  std::cout<<"Average deterministic sum many time  = "<<(time_deterministic_many.total/TESTS)<<std::endl;
  std::cout<<"Average deterministic sum manyc time = "<<(time_deterministic_manyc.total/TESTS)<<std::endl;
  std::cout<<"Average simple summation time        = "<<(time_simple.total/TESTS)<<std::endl;
  std::cout<<"Average Kahan summation time         = "<<(time_kahan.total/TESTS)<<std::endl;
  std::cout<<"Ratio Deterministic 1ata to Simple   = "<<(time_deterministic_1.total/time_simple.total)<<std::endl;
  std::cout<<"Ratio Deterministic 1ata to Kahan    = "<<(time_deterministic_1.total/time_kahan.total)<<std::endl;
  std::cout<<"Ratio Deterministic many to Simple   = "<<(time_deterministic_many.total/time_simple.total)<<std::endl;
  std::cout<<"Ratio Deterministic many to Kahan    = "<<(time_deterministic_many.total/time_kahan.total)<<std::endl;
  std::cout<<"Ratio Deterministic manyc to Simple  = "<<(time_deterministic_manyc.total/time_simple.total)<<std::endl;
  std::cout<<"Ratio Deterministic manyc to Kahan   = "<<(time_deterministic_manyc.total/time_kahan.total)<<std::endl;

  std::cout<<"Error bound                          = "<<ReproducibleFloatingAccumulator<FloatType>::error_bound(floats.size(), 1000, ref_val)<<std::endl;

  std::cout<<"Reference value                      = "<<std::fixed<<ref_val<<std::endl;
  std::cout<<"Reference bits                       = "<<binrep<FloatType>(ref_val)<<std::endl;

  std::cout<<"Distinct Kahan values                = "<<kahan_sums.size()<<std::endl;
  std::cout<<"Distinct Simple values               = "<<simple_sums.size()<<std::endl;

  for(const auto &kv: kahan_sums){
    std::cout<<"Kahan sum values (N="<<std::fixed<<kv.second<<") "<<kv.first<<" ("<<binrep<FloatType>(kv.first)<<")"<<std::endl;
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
  std::uniform_real_distribution<FloatType> distr(-1000, 1000);
  std::vector<FloatType> floats;
  for(int i=0;i<N;i++){
    floats.push_back(distr(gen));
  }
  PerformTestsOnData<FloatType, SimpleAccumType>(TESTS, floats, gen);
}



int main(){
  const int N = 1'000'000;
  const int TESTS = 100;

  PerformTests<float, float>(N, TESTS);
  PerformTests<double, double>(N, TESTS);

  // std::mt19937 gen(123456789);
  // std::uniform_real_distribution<double> distr(-1000, 1000);
  // std::vector<double> floats;
  // for(int i=0;i<N;i++){
  //   floats.push_back(distr(gen));
  // }

  // Timer time;
  // time.start();
  // ReproducibleFloatingAccumulator<double> rfa;
  // for(const auto &x: floats){
  //   rfa += x;
  // }
  // time.stop();

  // std::cout<<"Time = "<<time.total<<std::endl;

  return 0;
}




//TODO
  //This routine was provided as a means of allowing the you to optimize your code.
  //After you have called #binned_smsupdate() on Y with the maximum absolute value
  //of all future elements you wish to deposit in Y, you can call #binned_smsdeposit()
  //to deposit a maximum of #binned_SBENDURANCE elements into Y before renormalizing
  //Y with #binned_smrenorm(). After any number of successive calls of #binned_smsdeposit()
  //on Y, you must renormalize Y with #binned_smrenorm() before using any other function on Y.