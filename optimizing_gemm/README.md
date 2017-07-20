# Optimizing BLIS

### (on x86 architectures at least)

This tutorial illustrates some basic and some not-so-basic optimization techniques in the context of optimizing the GEMM (GEneral Matrix-Matrix multiply) routine from the Basic Linear Algebra Subprograms (BLAS). C++11 is used for the implementation.

The particular form of GEMM that will be implemented is the operation:

`C += A*B`

for general matrices `A`, `B`, and `C`. It is assumed (and enforced in the example program) that the matrices are row-major. The Eigen library is used to simplify some of the high-level handling of matrices, and to provide a check on the correctness and performance of the implementation (the Eigen library uses an external BLAS library, assumed to be OpenBLAS in the default Makefile).

## Step 0: Set-up and the triple loop

Prerequisites:
 - Unix or Linux OS (including OS X/macOS and Cygwin)
 - g++ (version 4.7 or later---another C++11 compiler may be used if set up in the Makefile)
 - OpenMP support (default in g++)
 - OpenBLAS (may also be changed in the Makefile)
 - Eigen v3
 - gnuplot
 - Intel or AMD x86-64 processor with AVX (for the last example only, FMA support is required)

The various example implementations of GEMM are contained in the files `my_dgemm_<n>.cxx` where `n` is from 0 to 8. To compile and automatically run and plot the first "triple-loop" example, run the command:

```
make STEP=0
```

Assuming everything went correctly, you should see a plot like this:

[[https://github.com/devinamatthews/molssi-sss-2017/blob/master/optimizing-gemm/figures/step0.png|alt=Step 0]]
