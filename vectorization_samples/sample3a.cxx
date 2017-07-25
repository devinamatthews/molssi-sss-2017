// transpose the column-major m*n matrix A
// into the column-major n*m matrix B
void transpose(int m, int n,
               const double* __restrict__ A, int lda,
                     double* __restrict__ B, int ldb)
{
    for (int i = 0;i < m;i++)
    {
        for (int j = 0;j < n;j++)
        {
            B[i*ldb + j] = A[i + j*lda];
        }
    }
}
