void dscal(int n, double alpha, double* x /* assume incx = 1 */)
{
    for (int i = 0;i < n;i++)
    {
        x[i] *= alpha;
    }
}
