Reproducible Floating-Point Summations
======================================

Floating-point sums are non-associative. That is,
```
(a+b)+c != a+(b+c)
```

This is problematic for performing reductions in which the ordering of the reduction or the partitioning of the data is non-deterministic, such as with internode communication or atomics.

This repo provides implementations of two papers which pre-round floating-point numbers to achieve reproducible results with (often) small reductions in accuracy versus a perfectly accurate summation (simulated in the test suite using Kahan summation). Both implementations' accuracy is likely to be greater than a simple summation.

The papers implemented are

 * Demmel and Nguyen (2013). "Fast Reproducible Floating-Point Summation".
 * Ahrens, Demmel, and Nguyen (2020). "Algorithms for Efficient Reproducible Floating Point Summation"



Building
---------------------------------------
```bash
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
```