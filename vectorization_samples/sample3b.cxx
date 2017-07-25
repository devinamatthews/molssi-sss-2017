// transpose the column-major m*n matrix A
// into the column-major n*m matrix B
void transpose(int m, int n,
               const double* __restrict__ A, int lda,
                     double* __restrict__ B, int ldb)
{
    // assume m%4 == 0 and n%4 == 0
    for (int i = 0;i < m;i += 4)
    {
        for (int j = 0;j < n;j += 4)
        {
            const double* __restrict__ Asub = &A[i + j*lda];
                  double* __restrict__ Bsub = &B[i*ldb + j];

             Bsub[0*ldb + 0] = Asub[0 + 0*lda];
             Bsub[0*ldb + 1] = Asub[0 + 1*lda];
             Bsub[0*ldb + 2] = Asub[0 + 2*lda];
             Bsub[0*ldb + 3] = Asub[0 + 3*lda];

             Bsub[1*ldb + 0] = Asub[1 + 0*lda];
             Bsub[1*ldb + 1] = Asub[1 + 1*lda];
             Bsub[1*ldb + 2] = Asub[1 + 2*lda];
             Bsub[1*ldb + 3] = Asub[1 + 3*lda];

             Bsub[2*ldb + 0] = Asub[2 + 0*lda];
             Bsub[2*ldb + 1] = Asub[2 + 1*lda];
             Bsub[2*ldb + 2] = Asub[2 + 2*lda];
             Bsub[2*ldb + 3] = Asub[2 + 3*lda];

             Bsub[3*ldb + 0] = Asub[3 + 0*lda];
             Bsub[3*ldb + 1] = Asub[3 + 1*lda];
             Bsub[3*ldb + 2] = Asub[3 + 2*lda];
             Bsub[3*ldb + 3] = Asub[3 + 3*lda];
        }
    }
}
