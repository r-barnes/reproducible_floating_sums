//g++ -O3 -g --std=c++17 repro_blas_extract_class.cpp -Wall
#include <algorithm>
#include <array>
#include <bitset>
#include <climits>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <math.h>
#include <random>
#include <unordered_map>
#include <vector>

extern "C"{
#include <binned.h>
#include <binnedBLAS.h>
#include <reproBLAS.h>
}

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



template<
  class ftype,
  int FOLD,
  typename std::enable_if<std::is_floating_point<ftype>::value>::type* = nullptr
>
class ReproducibleFloatingAccumulator {
 private:
  std::array<ftype, 2*FOLD> data = {0};

  static constexpr auto BIN_WIDTH = std::is_same<ftype, double>::value ? 40 : 13;
  static constexpr auto MIN_EXP = std::numeric_limits<ftype>::min_exponent;
  static constexpr auto MAX_EXP = std::numeric_limits<ftype>::max_exponent;
  static constexpr auto MANT_DIG = std::numeric_limits<ftype>::digits;
  static constexpr auto MAXINDEX = ((MAX_EXP - MIN_EXP + MANT_DIG - 1) / BIN_WIDTH) - 1;
  static constexpr auto MAXFOLD = MAXINDEX + 1;
  static constexpr auto COMPRESSION = 1.0 / (1 << (MANT_DIG - BIN_WIDTH + 1));
  static constexpr auto EXPANSION = 1.0 * (1 << (MANT_DIG - BIN_WIDTH + 1));
  static constexpr auto EXP_BIAS = MAX_EXP - 2;
  static constexpr auto EPSILON = std::numeric_limits<ftype>::epsilon();
  static constexpr auto ENDURANCE = 1 << (MANT_DIG - BIN_WIDTH - 2);

  static constexpr std::array<ftype, MAXINDEX + MAXFOLD> initializes_bins(){ //checked
    std::array<ftype, MAXINDEX + MAXFOLD> bins{0};

    if(std::is_same<ftype, float>::value){
      bins[0] = ldexpf(0.75, MAX_EXP);
    } else {
      bins[0] = 2.0 * ldexp(0.75, MAX_EXP - 1);
    }

    for(int index = 1; index <= MAXINDEX; index++){
      bins[index] = ldexp(0.75, MAX_EXP + MANT_DIG - BIN_WIDTH + 1 - index * BIN_WIDTH);
    }
    for(int index = MAXINDEX + 1; index < MAXINDEX + MAXFOLD; index++){
      bins[index] = bins[index - 1];
    }

    return bins;
  }

  static constexpr auto bins = initializes_bins();

  static inline constexpr const ftype* binned_bins(const int x) {
    return &bins[x];
  }

  static inline uint32_t& get_bits(float &x)       { return *reinterpret_cast<      uint32_t*>(&x);}
  static inline uint64_t& get_bits(double &x)      { return *reinterpret_cast<      uint64_t*>(&x);}
  static inline uint32_t  get_bits(const float &x) { return *reinterpret_cast<const uint32_t*>(&x);}
  static inline uint64_t  get_bits(const double &x){ return *reinterpret_cast<const uint64_t*>(&x);}

  static inline constexpr int ISNANINF(const ftype x) { //checked
    const auto bits = get_bits(x);
    return (bits & ((2ull * MAX_EXP - 1) << (MANT_DIG - 1))) == ((2ull * MAX_EXP - 1) << (MANT_DIG - 1));
  }

  static inline constexpr int EXP(const ftype x) { //checked
    const auto bits = get_bits(x);
    return (bits >> (MANT_DIG - 1)) & (2 * MAX_EXP - 1);
  }

  static inline constexpr int binned_dindex(const ftype x){ //checked
    int exp = EXP(x);
    if(exp == 0){
      if(x == 0.0){
        return MAXINDEX;
      } else {
        frexp(x, &exp);
        return std::min((MAX_EXP - exp)/BIN_WIDTH, MAXINDEX);
      }
    }
    return ((MAX_EXP + EXP_BIAS) - exp)/BIN_WIDTH;
  }

  inline ftype*       pvec()       { return &data[0];    }
  inline ftype*       cvec()       { return &data[FOLD]; }
  inline const ftype* pvec() const { return &data[0];    }
  inline const ftype* cvec() const { return &data[FOLD]; }

  inline int binned_index() const { //checked
    return ((MAX_EXP + MANT_DIG - BIN_WIDTH + 1 + EXP_BIAS) - EXP(pvec()[0]))/BIN_WIDTH;
  }

  inline bool binned_index0() const { //checked
    return EXP(pvec()[0]) == MAX_EXP + EXP_BIAS;
  }



/**
 * @internal
 * @brief Update manually specified binned double precision with double precision (X -> Y)
 *
 * This method updates Y to an index suitable for adding numbers with absolute value less than X
 *
 * @param fold the fold of the binned types
 * @param X scalar X
 * @param priY Y's primary vector
 * @param incpriY stride within Y's primary vector (use every incpriY'th element)
 * @param carY Y's carry vector
 * @param inccarY stride within Y's carry vector (use every inccarY'th element)
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   5 May 2015
 */
  void binned_dmdupdate(const ftype X, const int incpriY, const int inccarY) { //checked
    int i;
    int j;
    int X_index;
    int shift;
    auto *const priY = pvec();
    auto *const carY = cvec();

    if (ISNANINF(priY[0])){
      return;
    }

    X_index = binned_dindex(X);
    if(priY[0] == 0.0){
      const ftype *const bins = binned_bins(X_index);
      for(i = 0; i < FOLD; i++){
        priY[i * incpriY] = bins[i];
        carY[i * inccarY] = 0.0;
      }
    }else{
      shift = binned_index() - X_index;
      if(shift > 0){
        for(i = FOLD - 1; i >= shift; i--){
          priY[i * incpriY] = priY[(i - shift) * incpriY];
          carY[i * inccarY] = carY[(i - shift) * inccarY];
        }
        const ftype *const bins = binned_bins(X_index);
        for(j = 0; j < i + 1; j++){
          priY[j * incpriY] = bins[j];
          carY[j * inccarY] = 0.0;
        }
      }
    }
  }


  /**
   * @internal
   * @brief  Add double precision to suitably binned manually specified binned double precision (Y += X)
   *
   * Performs the operation Y += X on an binned type Y where the index of Y is larger than the index of X
   *
   * @note This routine was provided as a means of allowing the you to optimize your code. After you have called #binned_dmdupdate() on Y with the maximum absolute value of all future elements you wish to deposit in Y, you can call #binned_dmddeposit() to deposit a maximum of #binned_DBENDURANCE elements into Y before renormalizing Y with #binned_dmrenorm(). After any number of successive calls of #binned_dmddeposit() on Y, you must renormalize Y with #binned_dmrenorm() before using any other function on Y.
   *
   * @param fold the fold of the binned types
   * @param X scalar X
   * @param priY Y's primary vector
   * @param incpriY stride within Y's primary vector (use every incpriY'th element)
   *
   * @author Hong Diep Nguyen
   * @author Peter Ahrens
   * @date   10 Jun 2015
   */
  void binned_dmddeposit(const ftype X, const int incpriY){ //checked
    ftype M;
    int i;
    ftype x = X;
    auto *const priY = pvec();

    if(ISNANINF(x) || ISNANINF(priY[0])){
      priY[0] += x;
      return;
    }

    if(binned_index0()){
      M = priY[0];
      ftype qd = x * COMPRESSION;
      auto& ql = get_bits(qd);
      ql |= 1;
      qd += M;
      priY[0] = qd;
      M -= qd;
      M *= EXPANSION * 0.5;
      x += M;
      x += M;
      for (i = 1; i < FOLD - 1; i++) {
        M = priY[i * incpriY];
        qd = x;
        ql |= 1;
        qd += M;
        priY[i * incpriY] = qd;
        M -= qd;
        x += M;
      }
      qd = x;
      ql |= 1;
      priY[i * incpriY] += qd;
    } else {
      ftype qd = x;
      auto& ql = get_bits(qd);
      for (i = 0; i < FOLD - 1; i++) {
        M = priY[i * incpriY];
        qd = x;
        ql |= 1;
        qd += M;
        priY[i * incpriY] = qd;
        M -= qd;
        x += M;
      }
      qd = x;
      ql |= 1;
      priY[i * incpriY] += qd;
    }
  }


  /**
   * @internal
   * @brief Renormalize manually specified binned double precision
   *
   * Renormalization keeps the primary vector within the necessary bins by shifting over to the carry vector
   *
   * @param fold the fold of the binned types
   * @param priX X's primary vector
   * @param incpriX stride within X's primary vector (use every incpriX'th element)
   * @param carX X's carry vector
   * @param inccarX stride within X's carry vector (use every inccarX'th element)
   *
   * @author Hong Diep Nguyen
   * @author Peter Ahrens
   * @date   23 Sep 2015
   */
  void binned_dmrenorm(const int incpriX, const int inccarX) { //checked
    auto *priX = pvec();
    auto *carX = cvec();

    if(priX[0] == 0.0 || ISNANINF(priX[0])){
      return;
    }

    for (int i = 0; i < FOLD; i++, priX += incpriX, carX += inccarX) {
      auto tmp_renormd = priX[0];
      auto& tmp_renorml = get_bits(tmp_renormd);

      carX[0] += (int)((tmp_renorml >> (MANT_DIG - 3)) & 3) - 2;

      tmp_renorml &= ~(1ull << (MANT_DIG - 3));
      tmp_renorml |= 1ull << (MANT_DIG - 2);
      priX[0] = tmp_renormd;
    }
  }

  /**
   * @internal
   * @brief  Add double precision to manually specified binned double precision (Y += X)
   *
   * Performs the operation Y += X on an binned type Y
   *
   * @param fold the fold of the binned types
   * @param X scalar X
   * @param priY Y's primary vector
   * @param incpriY stride within Y's primary vector (use every incpriY'th element)
   * @param carY Y's carry vector
   * @param inccarY stride within Y's carry vector (use every inccarY'th element)
   *
   * @author Hong Diep Nguyen
   * @author Peter Ahrens
   * @date   27 Apr 2015
   */
  void binned_dmdadd(const ftype X, const int incpriY, const int inccarY){ //checked
    binned_dmdupdate(X, incpriY, inccarY);
    binned_dmddeposit(X, incpriY);
    binned_dmrenorm(incpriY, inccarY);
  }

  /**
   * @internal
   * @brief Convert manually specified binned double precision to double precision (X -> Y)
   *
   * @param fold the fold of the binned types
   * @param priX X's primary vector
   * @param incpriX stride within X's primary vector (use every incpriX'th element)
   * @param carX X's carry vector
   * @param inccarX stride within X's carry vector (use every inccarX'th element)
   * @return scalar Y
   *
   * @author Peter Ahrens
   * @date   31 Jul 2015
   */
  double binned_conv_double(const int incpriX, const int inccarX) const {
    int i = 0;

    const auto *const priX = pvec();
    const auto *const carX = cvec();

    if (ISNANINF(priX[0])){
      return priX[0];
    }

    if (priX[0] == 0.0) {
      return 0.0;
    }

    double Y = 0.0;
    double scale_down;
    double scale_up;
    int scaled;
    const auto X_index = binned_index();
    const auto *const bins = binned_bins(X_index);
    if(X_index <= (3 * MANT_DIG)/BIN_WIDTH){
      scale_down = ldexp(0.5, 1 - (2 * MANT_DIG - BIN_WIDTH));
      scale_up = ldexp(0.5, 1 + (2 * MANT_DIG - BIN_WIDTH));
      scaled = std::max(std::min(FOLD, (3 * MANT_DIG)/BIN_WIDTH - X_index), 0);
      if(X_index == 0){
        Y += carX[0] * ((bins[0]/6.0) * scale_down * EXPANSION);
        Y += carX[inccarX] * ((bins[1]/6.0) * scale_down);
        Y += (priX[0] - bins[0]) * scale_down * EXPANSION;
        i = 2;
      }else{
        Y += carX[0] * ((bins[0]/6.0) * scale_down);
        i = 1;
      }
      for(; i < scaled; i++){
        Y += carX[i * inccarX] * ((bins[i]/6.0) * scale_down);
        Y += (priX[(i - 1) * incpriX] - bins[i - 1]) * scale_down;
      }
      if(i == FOLD){
        Y += (priX[(FOLD - 1) * incpriX] - bins[FOLD - 1]) * scale_down;
        return Y * scale_up;
      }
      if(isinf(Y * scale_up)){
        return Y * scale_up;
      }
      Y *= scale_up;
      for(; i < FOLD; i++){
        Y += carX[i * inccarX] * (bins[i]/6.0);
        Y += priX[(i - 1) * incpriX] - bins[i - 1];
      }
      Y += priX[(FOLD - 1) * incpriX] - bins[FOLD - 1];
    }else{
      Y += carX[0] * (bins[0]/6.0);
      for(i = 1; i < FOLD; i++){
        Y += carX[i * inccarX] * (bins[i]/6.0);
        Y += (priX[(i - 1) * incpriX] - bins[i - 1]);
      }
      Y += (priX[(FOLD - 1) * incpriX] - bins[FOLD - 1]);
    }
    return Y;
  }

  /**
   * @internal
   * @brief Convert manually specified binned single precision to single precision (X -> Y)
   *
   * @param fold the fold of the binned types
   * @param priX X's primary vector
   * @param incpriX stride within X's primary vector (use every incpriX'th element)
   * @param carX X's carry vector
   * @param inccarX stride within X's carry vector (use every inccarX'th element)
   * @return scalar Y
   *
   * @author Hong Diep Nguyen
   * @author Peter Ahrens
   * @date   27 Apr 2015
  */
  float binned_conv_single(const int incpriX, const int inccarX) const { //checked
    int i = 0;
    double Y = 0.0;
    const auto *const priX = pvec();
    const auto *const carX = cvec();

    if (ISNANINF(priX[0])){
      return priX[0];
    }

    if (priX[0] == 0.0) {
      return 0.0;
    }

    //Note that the following order of summation is in order of decreasing
    //exponent. The following code is specific to SBWIDTH=13, FLT_MANT_DIG=24, and
    //the number of carries equal to 1.
    const auto X_index = binned_index();
    const auto *const bins = binned_bins(X_index);
    if(X_index == 0){
      Y += (double)carX[0] * (double)(bins[0]/6.0) * (double)EXPANSION;
      Y += (double)carX[inccarX] * (double)(bins[1]/6.0);
      Y += (double)(priX[0] - bins[0]) * (double)EXPANSION;
      i = 2;
    }else{
      Y += (double)carX[0] * (double)(bins[0]/6.0);
      i = 1;
    }
    for(; i < FOLD; i++){
      Y += (double)carX[i * inccarX] * (double)(bins[i]/6.0);
      Y += (double)(priX[(i - 1) * incpriX] - bins[i - 1]);
    }
    Y += (double)(priX[(FOLD - 1) * incpriX] - bins[FOLD - 1]);

    return (float)Y;
  }

  /**
   * @internal
   * @brief  Add manually specified binned double precision (Y += X)
   *
   * Performs the operation Y += X
   *
   * @param fold the fold of the binned types
   * @param priX X's primary vector
   * @param incpriX stride within X's primary vector (use every incpriX'th element)
   * @param carX X's carry vector
   * @param inccarX stride within X's carry vector (use every inccarX'th element)
   * @param priY Y's primary vector
   * @param incpriY stride within Y's primary vector (use every incpriY'th element)
   * @param carY Y's carry vector
   * @param inccarY stride within Y's carry vector (use every inccarY'th element)
   *
   * @author Hong Diep Nguyen
   * @author Peter Ahrens
   * @date   27 Apr 2015
   */
  void binned_dmdmadd(const ReproducibleFloatingAccumulator &other, const int incpriX, const int inccarX, const int incpriY, const int inccarY) {
    auto *const priX = pvec();
    auto *const carX = cvec();
    auto *const priY = other.pvec();
    auto *const carY = other.cvec();

    if (priX[0] == 0.0)
      return;

    if (priY[0] == 0.0) {
      for (int i = 0; i < FOLD; i++) {
        priY[i*incpriY] = priX[i*incpriX];
        carY[i*inccarY] = carX[i*inccarX];
      }
      return;
    }

    if (ISNANINF(priX[0]) || ISNANINF(priY[0])){
      priY[0] += priX[0];
      return;
    }

    const auto X_index = binned_index(priX);
    const auto Y_index = binned_index(priY);
    const auto shift = Y_index - X_index;
    if(shift > 0){
      const auto *const bins = binned_bins(Y_index);
      //shift Y upwards and add X to Y
      for (int i = FOLD - 1; i >= shift; i--) {
        priY[i*incpriY] = priX[i*incpriX] + (priY[(i - shift)*incpriY] - bins[i - shift]);
        carY[i*inccarY] = carX[i*inccarX] + carY[(i - shift)*inccarY];
      }
      for (int i = 0; i < shift && i < FOLD; i++) {
        priY[i*incpriY] = priX[i*incpriX];
        carY[i*inccarY] = carX[i*inccarX];
      }
    }else{
      const auto *const bins = binned_bins(X_index);
      //shift X upwards and add X to Y
      for (int i = 0 - shift; i < FOLD; i++) {
        priY[i*incpriY] += priX[(i + shift)*incpriX] - bins[i + shift];
        carY[i*inccarY] += carX[(i + shift)*inccarX];
      }
    }

    binned_dmrenorm(incpriY, inccarY);
  }

  void binned_dbdbadd(const ReproducibleFloatingAccumulator &other){
    binned_dmdmadd(other, 1, 1, 1, 1);
  }


 public:
  void zero() {
    data = {0};
  }

  void binned_dbdadd(const ftype X){
    binned_dmdadd(X, 1, 1);
  }

  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value>::type* = nullptr>
  ReproducibleFloatingAccumulator& operator+=(const U x){
    binned_dmdadd(static_cast<ftype>(x), 1, 1);
    return *this;
  }

  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value>::type* = nullptr>
  ReproducibleFloatingAccumulator& operator-=(const U x){
    binned_dmdadd(-static_cast<ftype>(x), 1, 1);
    return *this;
  }

  ReproducibleFloatingAccumulator& operator+=(const ReproducibleFloatingAccumulator &other){
    binned_dbdbadd(other);
    return *this;
  }

  ReproducibleFloatingAccumulator& operator-=(const ReproducibleFloatingAccumulator &other){
    throw std::runtime_error("Not implemented!");
    // binned_dbdbadd(other);
    // return *this;
  }

  bool operator==(const ReproducibleFloatingAccumulator &other) const {
    return data==other.data;
  }

  bool operator!=(const ReproducibleFloatingAccumulator &other) const {
    return !operator==(other);
  }

  template <typename U, typename std::enable_if<std::is_arithmetic<U>::value>::type* = nullptr>
  ReproducibleFloatingAccumulator& operator=(const U x){
    zero();
    binned_dmdadd(static_cast<ftype>(x), 1, 1);
    return *this;
  }

  ReproducibleFloatingAccumulator& operator=(const ReproducibleFloatingAccumulator<ftype, FOLD> &o){
    data = o.data;
    return *this;
  }

  ReproducibleFloatingAccumulator operator-() {
    constexpr int incpriX = 1;
    constexpr int inccarX = 1;
    ReproducibleFloatingAccumulator temp = *this;
    if(pvec()[0] != 0.0){
      const auto *const bins = binned_bins(binned_index());
      for (int i = 0; i < FOLD; i++) {
        temp.pvec()[i * incpriX] = bins[i] - (pvec()[i * incpriX] - bins[i]);
        temp.cvec()[i * inccarX] = -cvec()[i * inccarX];
      }
    }
    return temp;
  }

  ftype conv() const {
    if(std::is_same<ftype, float>::value){
      return binned_conv_single(1, 1);
    } else {
      return binned_conv_double(1, 1);
    }
  }

  /**
   * @brief Get binned single precision summation error bound
   *
   * This is a bound on the absolute error of a summation using binned types
   *
   * @param N the number of single precision floating point summands
   * @param X the summand of maximum absolute value
   * @param S the value of the sum computed using binned types
   * @return error bound
   */
  static constexpr ftype error_bound(const uint64_t N, const ftype X, const ftype S) {
    const double Xval = std::abs(X);
    const double Sval = std::abs(S);
    return static_cast<ftype>(std::max(Xval, ldexp(0.5, MIN_EXP - 1)) * ldexp(0.5, (1 - FOLD) * BIN_WIDTH + 1) * N + ((7.0 * EPSILON) / (1.0 - 6.0 * std::sqrt(static_cast<double>(EPSILON)) - 7.0 * EPSILON)) * Sval);
  }

  //This routine was provided as a means of allowing the you to optimize your code.
  //After you have called #binned_smsupdate() on Y with the maximum absolute value
  //of all future elements you wish to deposit in Y, you can call #binned_smsdeposit()
  //to deposit a maximum of #binned_SBENDURANCE elements into Y before renormalizing
  //Y with #binned_smrenorm(). After any number of successive calls of #binned_smsdeposit()
  //on Y, you must renormalize Y with #binned_smrenorm() before using any other function on Y.
  template <typename InputIt>
  void add(InputIt first, InputIt last, const ftype max_abs_val) {
    binned_dmdupdate(std::abs(max_abs_val), 1, 1);
    size_t count = 0;
    for(;first!=last;first++,count++){
      binned_dmddeposit(static_cast<ftype>(*first), 1);
      if(count==ENDURANCE){
        binned_dmrenorm(1, 1);
        count = 0;
      }
    }
  }

  template <typename InputIt>
  void add(InputIt first, InputIt last) {
    const auto max_abs_val = *std::max_element(first, last, [](const auto &a, const auto &b){
      return std::abs(a) < std::abs(b);
    });
    add(first, last, static_cast<ftype>(max_abs_val));
  }

  template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
  void add(const T *input, const size_t N, const ftype max_abs_val) {
    if(N==0){
      return;
    }
    add(input, input + N, max_abs_val);
  }

  template <typename T, typename std::enable_if<std::is_arithmetic<T>::value>::type* = nullptr>
  void add(const T *input, const size_t N) {
    if(N==0){
      return;
    }
    T max_abs_val = input[0];
    for(size_t i=0;i<N;i++){
      max_abs_val = std::max(max_abs_val, std::abs(input[i]));
    }
    add(input, N, max_abs_val);
  }
};


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



template<class FloatType>
FloatType bitwise_deterministic_summation_1(const std::vector<FloatType> &vec){
  ReproducibleFloatingAccumulator<FloatType, 3> rfa;
  for(const auto &x: vec){
    // rfa.binned_dbdadd(x);
    rfa += x;
  }
  return rfa.conv();
}

template<class FloatType>
FloatType bitwise_deterministic_summation_many(const std::vector<FloatType> &vec){
  ReproducibleFloatingAccumulator<FloatType, 3> rfa;
  rfa.add(vec.begin(), vec.end());
  return rfa.conv();
}

template<class FloatType>
FloatType bitwise_deterministic_summation_manyc(const std::vector<FloatType> &vec, const FloatType max_abs_val){
  ReproducibleFloatingAccumulator<FloatType, 3> rfa;
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

  std::cout<<"Error bound                          = "<<ReproducibleFloatingAccumulator<FloatType, 3>::error_bound(floats.size(), 1000, ref_val)<<std::endl;

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
  // ReproducibleFloatingAccumulator<double, 3> rfa;
  // for(const auto &x: floats){
  //   rfa += x;
  // }
  // time.stop();

  // std::cout<<"Time = "<<time.total<<std::endl;

  return 0;
}
