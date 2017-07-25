void daxpy(int n, double alpha,
           const double* x, /* assume incx = 1 */
                 double* y  /* assume incy = 1 */)
{
    for (int i = 0;i < n;i++)
    {
        y[i] += alpha*x[i];
    }
}
