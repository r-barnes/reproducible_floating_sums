#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <math.h>
#include <vector>

typedef union long_double_ {
  double   d;
  uint64_t l;
} long_double;

#define FLT_MAX_EXP __FLT_MAX_EXP__
#define DBL_MAX_EXP __DBL_MAX_EXP__
#define LDBL_MAX_EXP __LDBL_MAX_EXP__

#define FLT_MIN_EXP __FLT_MIN_EXP__
#define DBL_MIN_EXP __DBL_MIN_EXP__
#define LDBL_MIN_EXP __LDBL_MIN_EXP__

#define FLT_MANT_DIG __FLT_MANT_DIG__
#define DBL_MANT_DIG __DBL_MANT_DIG__
#define LDBL_MANT_DIG __LDBL_MANT_DIG__

#ifndef MAX
#define MAX(A,B) ((A)>(B)?(A):(B))
#endif

#ifndef MIN
#define MIN(A,B) ((A)<(B)?(A):(B))
#endif

/**
 * @brief Binned double precision bin width
 *
 * bin width (in bits)
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
#define DBWIDTH 40

/**
 * @brief Binned double precision maximum index
 *
 * maximum index (inclusive)
 *
 * @author Peter Ahrens
 * @date   24 Jun 2015
 */
#define binned_DBMAXINDEX (((DBL_MAX_EXP - DBL_MIN_EXP + DBL_MANT_DIG - 1)/DBWIDTH) - 1)

inline int ISNANINF(double X) {
  long_double tmp_ISNANINF;
  tmp_ISNANINF.d = X;
  return (tmp_ISNANINF.l & ((2ull * DBL_MAX_EXP - 1) << (DBL_MANT_DIG - 1))) == ((2ull * DBL_MAX_EXP - 1) << (DBL_MANT_DIG - 1));
}

/**
 * @brief Binned double precision maximum index
 *
 * maximum index (inclusive)
 *
 * @author Peter Ahrens
 * @date   24 Jun 2015
 */
#define binned_DBMAXINDEX (((DBL_MAX_EXP - DBL_MIN_EXP + DBL_MANT_DIG - 1)/DBWIDTH) - 1)

#define FLT_MANT_DIG __FLT_MANT_DIG__
#define DBL_MANT_DIG __DBL_MANT_DIG__
#define LDBL_MANT_DIG __LDBL_MANT_DIG__

#define EXP_BIAS (DBL_MAX_EXP - 2)

inline int EXP(double X) {
  long_double tmp_EXP;
  tmp_EXP.d = X;
  return (tmp_EXP.l >> (DBL_MANT_DIG - 1)) & (2 * DBL_MAX_EXP - 1);
}

/**
 * @brief The binned double datatype
 *
 * To allocate a #double_binned, call binned_dballoc()
 *
 * @warning A #double_binned is, under the hood, an array of @c double. Therefore, if you have defined an array of #double_binned, you must index it by multiplying the index into the array by the number of underlying @c double that make up the #double_binned. This number can be obtained by a call to binned_dbnum()
 */
typedef double double_binned;

/**
 * @brief binned double precision size
 *
 * @param fold the fold of the binned type
 * @return the size (in bytes) of the binned type
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
size_t binned_dbsize(const int fold){
  return 2*fold*sizeof(double);
}

/**
 * @brief binned double precision allocation
 *
 * @param fold the fold of the binned type
 * @return a freshly allocated binned type. (free with @c free())
 *
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
double_binned *binned_dballoc(const int fold){
  return (double_binned*)malloc(binned_dbsize(fold));
}

/**
 * @brief Binned double precision maximum index
 *
 * maximum index (inclusive)
 *
 * @author Peter Ahrens
 * @date   24 Jun 2015
 */
#define binned_DBMAXINDEX (((DBL_MAX_EXP - DBL_MIN_EXP + DBL_MANT_DIG - 1)/DBWIDTH) - 1)


/**
 * @brief The maximum double precision fold supported by the library.
 *
 * @author Peter Ahrens
 * @date   14 Jan 2016
 */
#define binned_DBMAXFOLD (binned_DBMAXINDEX + 1)


static double bins[binned_DBMAXINDEX + binned_DBMAXFOLD];
static int bins_initialized = 0;

/**
 * @internal
 * @brief Get binned double precision reference bins
 *
 * returns a pointer to the bins corresponding to the given index
 *
 * @param X index
 * @return pointer to constant double precision bins of index X
 *
 * @author Peter Ahrens
 * @author Hong Diep Nguyen
 * @date   19 Jun 2015
 */
const double *binned_dmbins(const int X) {
  int index;

  if (!bins_initialized) {
    bins[0] = 2.0 * ldexp(0.75, DBL_MAX_EXP - 1);
    for(index = 1; index <= binned_DBMAXINDEX; index++){
      bins[index] = ldexp(0.75, (DBL_MAX_EXP + DBL_MANT_DIG - DBWIDTH + 1 - index * DBWIDTH));
    }
    for(; index < binned_DBMAXINDEX + binned_DBMAXFOLD; index++){
      bins[index] = bins[index - 1];
    }

    bins_initialized = 1;
  }

  return (const double*)bins + X;
}

/**
 * @brief Set binned double precision to 0 (X = 0)
 *
 * Performs the operation X = 0
 *
 * @param fold the fold of the binned types
 * @param X binned scalar X
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
void binned_dbsetzero(const int fold, double_binned *X){
  memset(X, 0, binned_dbsize(fold));
}

/**
 * @brief Get index of double precision
 *
 * The index of a non-binned type is the smallest index an binned type would need to have to sum it reproducibly. Higher indicies correspond to smaller bins.
 *
 * @param X scalar X
 * @return X's index
 *
 * @author Peter Ahrens
 * @author Hong Diep Nguyen
 * @date   19 Jun 2015
 */
int binned_dindex(const double X){
  /*
  //reference version
  int exp;

  if(X == 0.0){
    return (DBL_MAX_EXP - DBL_MIN_EXP)/DBWIDTH;
  }else{
    frexp(X, &exp);
    return (DBL_MAX_EXP - exp)/DBWIDTH;
  }
  */
  int exp = EXP(X);
  if(exp == 0){
    if(X == 0.0){
      return binned_DBMAXINDEX;
    }else{
      frexp(X, &exp);
      return MIN((DBL_MAX_EXP - exp)/DBWIDTH, binned_DBMAXINDEX);
    }
  }
  return ((DBL_MAX_EXP + EXP_BIAS) - exp)/DBWIDTH;
}

/**
 * @internal
 * @brief Get index of manually specified binned double precision
 *
 * The index of an binned type is the bin that it corresponds to. Higher indicies correspond to smaller bins.
 *
 * @param priX X's primary vector
 * @return X's index
 *
 * @author Peter Ahrens
 * @author Hong Diep Nguyen
 * @date   23 Sep 2015
 */
int binned_dmindex(const double *priX){
  /*
  //reference version
  int exp;

  if(priX[0] == 0.0){
    return (DBL_MAX_EXP - DBL_MIN_EXP)/DBWIDTH + binned_DBMAXFOLD;
  }else{
    frexp(priX[0], &exp);
    if(exp == DBL_MAX_EXP){
      return 0;
    }
    return (DBL_MAX_EXP + DBL_MANT_DIG - DBWIDTH + 1 - exp)/DBWIDTH;
  }
  */
  return ((DBL_MAX_EXP + DBL_MANT_DIG - DBWIDTH + 1 + EXP_BIAS) - EXP(priX[0]))/DBWIDTH;
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
void binned_dmdupdate(const int fold, const double X, double* priY, const int incpriY, double* carY, const int inccarY) {
  int i;
  int j;
  int X_index;
  int shift;
  const double *bins;

  if (ISNANINF(priY[0])){
    return;
  }

  X_index = binned_dindex(X);
  if(priY[0] == 0.0){
    bins = binned_dmbins(X_index);
    for(i = 0; i < fold; i++){
      priY[i * incpriY] = bins[i];
      carY[i * inccarY] = 0.0;
    }
  }else{
    shift = binned_dmindex(priY) - X_index;
    if(shift > 0){
      for(i = fold - 1; i >= shift; i--){
        priY[i * incpriY] = priY[(i - shift) * incpriY];
        carY[i * inccarY] = carY[(i - shift) * inccarY];
      }
      bins = binned_dmbins(X_index);
      for(j = 0; j < i + 1; j++){
        priY[j * incpriY] = bins[j];
        carY[j * inccarY] = 0.0;
      }
    }
  }
}

/**
 * @internal
 * @brief Check if index of manually specified binned double precision is 0
 *
 * A quick check to determine if the index is 0
 *
 * @param priX X's primary vector
 * @return >0 if x has index 0, 0 otherwise.
 *
 * @author Peter Ahrens
 * @date   19 May 2015
 */
int binned_dmindex0(const double *priX){
  /*
  //reference version
  int exp;

  frexp(priX[0], &exp);
  if(exp == DBL_MAX_EXP){
    return 1;
  }
  return 0;
  */
  return EXP(priX[0]) == DBL_MAX_EXP + EXP_BIAS;
}

/**
 * @internal
 * @brief Binned double precision compression factor
 *
 * This factor is used to scale down inputs before deposition into the bin of highest index
 *
 * @author Peter Ahrens
 * @date   19 May 2015
 */
#define binned_DMCOMPRESSION (1.0/(1 << (DBL_MANT_DIG - DBWIDTH + 1)))

/**
 * @internal
 * @brief Binned double precision expansion factor
 *
 * This factor is used to scale up inputs after deposition into the bin of highest index
 *
 * @author Peter Ahrens
 * @date   19 May 2015
 */
#define binned_DMEXPANSION (1.0*(1 << (DBL_MANT_DIG - DBWIDTH + 1)))


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
void binned_dmddeposit(const int fold, const double X, double *priY, const int incpriY){
  double M;
  long_double q;
  int i;
  double x = X;

  if(ISNANINF(x) || ISNANINF(priY[0])){
    priY[0] += x;
    return;
  }

  if(binned_dmindex0(priY)){
    M = priY[0];
    q.d = x * binned_DMCOMPRESSION;
    q.l |= 1;
    q.d += M;
    priY[0] = q.d;
    M -= q.d;
    M *= binned_DMEXPANSION * 0.5;
    x += M;
    x += M;
    for (i = 1; i < fold - 1; i++) {
      M = priY[i * incpriY];
      q.d = x;
      q.l |= 1;
      q.d += M;
      priY[i * incpriY] = q.d;
      M -= q.d;
      x += M;
    }
    q.d = x;
    q.l |= 1;
    priY[i * incpriY] += q.d;
  }else{
    for (i = 0; i < fold - 1; i++) {
      M = priY[i * incpriY];
      q.d = x;
      q.l |= 1;
      q.d += M;
      priY[i * incpriY] = q.d;
      M -= q.d;
      x += M;
    }
    q.d = x;
    q.l |= 1;
    priY[i * incpriY] += q.d;
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
void binned_dmrenorm(const int fold, double* priX, const int incpriX, double* carX, const int inccarX) {
  /*
  //reference version
  int i;
  double M;
  double priX0 = priX[0];

  if(priX0 == 0.0 || ISNANINF(priX0)){
    return;
  }

  for (i = 0; i < fold; i++, priX += incpriX, carX += inccarX) {
    priX0 = priX[0];

    M = UFP(priX0);

    if (priX0 >= (M * 1.75)) {
      priX[0] -= M * 0.25;
      carX[0] += 1;
    }
    else if (priX0 < (M * 1.5)) {
      priX[0] += M * 0.25;
      carX[0] -= 1;
    }
  }
  */
  /*
  //vectorizeable version
  int i;
  long_double tmp_renorm, tmp_c;
  long tmp;

  for (i = 0; i < fold; i++, priX += incpriX, carX += inccarX) {
    tmp_renorm.d = priX[0];
    tmp_c.d = priX[0];

    tmp_c.l &= ((1ull << (DBL_MANT_DIG - 3)) | (1ull << (DBL_MANT_DIG - 2)));
    tmp_c.l <<= (65 - DBL_MANT_DIG);
    carX[0] -= 0.5 * tmp_c.d;

    tmp = tmp_renorm.l & (1ull << (DBL_MANT_DIG - 3));
    tmp <<= 1;
    tmp_renorm.l |= tmp;
    tmp_renorm.l &= ~(1ull << (DBL_MANT_DIG - 3));
    priX[0] = tmp_renorm.d;
  }
  */
  int i;
  long_double tmp_renorm;

  if(priX[0] == 0.0 || ISNANINF(priX[0])){
    return;
  }

  for (i = 0; i < fold; i++, priX += incpriX, carX += inccarX) {
    tmp_renorm.d = priX[0];

    carX[0] += (int)((tmp_renorm.l >> (DBL_MANT_DIG - 3)) & 3) - 2;

    tmp_renorm.l &= ~(1ull << (DBL_MANT_DIG - 3));
    tmp_renorm.l |= 1ull << (DBL_MANT_DIG - 2);
    priX[0] = tmp_renorm.d;
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
void binned_dmdadd(const int fold, const double X, double *priY, const int incpriY, double *carY, const int inccarY){
  binned_dmdupdate(fold, X, priY, incpriY, carY, inccarY);
  binned_dmddeposit(fold, X, priY, incpriY);
  binned_dmrenorm(fold, priY, incpriY, carY, inccarY);
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
double binned_ddmconv(const int fold, const double* priX, const int incpriX, const double* carX, const int inccarX) {
  int i = 0;
  int X_index;
  const double *bins;

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
  X_index = binned_dmindex(priX);
  bins = binned_dmbins(X_index);
  if(X_index <= (3 * DBL_MANT_DIG)/DBWIDTH){
    scale_down = ldexp(0.5, 1 - (2 * DBL_MANT_DIG - DBWIDTH));
    scale_up = ldexp(0.5, 1 + (2 * DBL_MANT_DIG - DBWIDTH));
    scaled = MAX(MIN(fold, (3 * DBL_MANT_DIG)/DBWIDTH - X_index), 0);
    if(X_index == 0){
      Y += carX[0] * ((bins[0]/6.0) * scale_down * binned_DMEXPANSION);
      Y += carX[inccarX] * ((bins[1]/6.0) * scale_down);
      Y += (priX[0] - bins[0]) * scale_down * binned_DMEXPANSION;
      i = 2;
    }else{
      Y += carX[0] * ((bins[0]/6.0) * scale_down);
      i = 1;
    }
    for(; i < scaled; i++){
      Y += carX[i * inccarX] * ((bins[i]/6.0) * scale_down);
      Y += (priX[(i - 1) * incpriX] - bins[i - 1]) * scale_down;
    }
    if(i == fold){
      Y += (priX[(fold - 1) * incpriX] - bins[fold - 1]) * scale_down;
      return Y * scale_up;
    }
    if(isinf(Y * scale_up)){
      return Y * scale_up;
    }
    Y *= scale_up;
    for(; i < fold; i++){
      Y += carX[i * inccarX] * (bins[i]/6.0);
      Y += priX[(i - 1) * incpriX] - bins[i - 1];
    }
    Y += priX[(fold - 1) * incpriX] - bins[fold - 1];
  }else{
    Y += carX[0] * (bins[0]/6.0);
    for(i = 1; i < fold; i++){
      Y += carX[i * inccarX] * (bins[i]/6.0);
      Y += (priX[(i - 1) * incpriX] - bins[i - 1]);
    }
    Y += (priX[(fold - 1) * incpriX] - bins[fold - 1]);
  }
  return Y;
}


/**
 * @brief Convert binned double precision to double precision (X -> Y)
 *
 * @param fold the fold of the binned types
 * @param X binned scalar X
 * @return scalar Y
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
double binned_ddbconv(const int fold, const double_binned *X) {
  return binned_ddmconv(fold, X, 1, X + fold, 1);
}


/**
 * @brief  Add double precision to binned double precision (Y += X)
 *
 * Performs the operation Y += X on an binned type Y
 *
 * @param fold the fold of the binned types
 * @param X scalar X
 * @param Y binned scalar Y
 *
 * @author Hong Diep Nguyen
 * @author Peter Ahrens
 * @date   27 Apr 2015
 */
void binned_dbdadd(const int fold, const double X, double_binned *Y){
  binned_dmdadd(fold, X, Y, 1, Y + fold, 1);
}

std::vector<double> sine_vector(int n){
  std::vector<double> x(n);

  // Set x to be a sine wave
  for(int i = 0; i < n; i++){
    x[i] = sin(2 * M_PI * (i / (double)n - 0.5));
  }

  return x;
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

int main(){
  const int n = 1'000'000;
  const std::vector<double> x = sine_vector(n);

  // Here, we sum x using binned primitives. This is less efficient than the
  // optimized reproBLAS_sum method, but might be useful if the data isn't
  // arranged in a vector.
  Timer bin_time;
  bin_time.start();
  double_binned *isum = binned_dballoc(3);
  binned_dbsetzero(3, isum);
  for(int i = 0; i < n; i++){
    binned_dbdadd(3, x[i], isum);
  }
  auto sum = binned_ddbconv(3, isum);
  bin_time.stop();

  Timer kahan_time;
  kahan_time.start();
  const auto check_sum = serial_kahan_summation<long double>(x);
  kahan_time.stop();

  std::cout<<"Binned sum "<<sum<<", time = "<<bin_time.total<<std::endl;
  std::cout<<"Checked sum "<<check_sum<<", time = "<<kahan_time.total<<std::endl;
  std::cout<<"Ratio (lower is better): "<<(bin_time.total/kahan_time.total)<<std::endl;
}
