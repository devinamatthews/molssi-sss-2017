#include <immintrin.h>

void transpose_4x4(__m256d A[4], __m256d B[4])
{
    __m256d tmp[4];
    // A[0] = (A00, A10, A20, A30)
    // A[1] = (A01, A11, A21, A31)
    // A[2] = (A02, A12, A22, A32)
    // A[3] = (A03, A13, A23, A33)
    tmp[0] = _mm256_shuffle_pd(A[0], A[1], 0x0);
    tmp[1] = _mm256_shuffle_pd(A[0], A[1], 0xf);
    tmp[2] = _mm256_shuffle_pd(A[2], A[3], 0x0);
    tmp[3] = _mm256_shuffle_pd(A[2], A[3], 0xf);
    // tmp[0] = (A00, A01, A20, A21)
    // tmp[1] = (A10, A11, A30, A31)
    // tmp[2] = (A02, A03, A22, A23)
    // tmp[3] = (A12, A13, A32, A33)
    B[0] = _mm256_permute2f128_pd(tmp[0], tmp[2], 0x20);
    B[1] = _mm256_permute2f128_pd(tmp[1], tmp[3], 0x20);
    B[2] = _mm256_permute2f128_pd(tmp[0], tmp[2], 0x31);
    B[3] = _mm256_permute2f128_pd(tmp[1], tmp[3], 0x31);
    // B[0] = (A00, A01, A02, A03)
    // B[1] = (A10, A11, A12, A13)
    // B[2] = (A20, A21, A22, A23)
    // B[3] = (A30, A31, A32, A33)
}

// transpose the column-major m*n matrix A
// into the column-major n*m matrix B
void transpose(int m, int n,
               const double* __restrict__ A, int lda,
                     double* __restrict__ B, int ldb)
{
    __m256d Areg[4], Breg[4];

    // assume m%4 == 0 and n%4 == 0
    for (int i = 0;i < m;i += 4)
    {
        for (int j = 0;j < n;j += 4)
        {
            const double* __restrict__ Asub = &A[i + j*lda];
                  double* __restrict__ Bsub = &B[i*ldb + j];

             Areg[0] = _mm256_loadu_pd(Asub + 0*lda);
             Areg[1] = _mm256_loadu_pd(Asub + 1*lda);
             Areg[2] = _mm256_loadu_pd(Asub + 2*lda);
             Areg[3] = _mm256_loadu_pd(Asub + 3*lda);

             transpose_4x4(Areg, Breg);

             _mm256_storeu_pd(Bsub + 0*ldb, Breg[0]);
             _mm256_storeu_pd(Bsub + 1*ldb, Breg[1]);
             _mm256_storeu_pd(Bsub + 2*ldb, Breg[2]);
             _mm256_storeu_pd(Bsub + 3*ldb, Breg[3]);
        }
    }
}
