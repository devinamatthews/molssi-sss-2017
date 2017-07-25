void daxpy(int n, double alpha,
           //
           // Use the __restrict__ keyword to promise to the compiler
           // that x and y don't overlap.
           //
           const double* __restrict__ x, /* assume incx = 1 */
                 double* __restrict__ y  /* assume incy = 1 */)
{
    // OR: use #pragmas to force vectorization:
    //
    // ivdep: ignore assumed data dependencies
    //
    #pragma GCC ivdep // gcc
    #pragma ivdep // icpc
    //
    // simd: always vectorize
    //
    #pragma simd //icpc
    #pragma omp simd //any compiler with OpenMP 4
    for (int i = 0;i < n;i++)
    {
        y[i] += alpha*x[i];
    }
}
