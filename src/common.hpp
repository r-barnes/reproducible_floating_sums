#pragma once

#include <bitset>               //Used for showing bitwise representations
#include <chrono>               //Used for timing algorithms
#include <climits>              //Used for showing bitwise representations
#include <ostream>              //Used for showing bitwise representations
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
template<class AccumType, class FloatType>
FloatType serial_simple_summation(const std::vector<FloatType> &vec){
  AccumType sum = 0;
  for(const auto &x: vec){
    sum += x;
  }
  return sum;
}