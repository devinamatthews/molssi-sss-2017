#include "common.hpp"

#include "immintrin.h"

#define M_BLOCK 72
#define N_BLOCK 4080
#define K_BLOCK 256

#define M_UNROLL 6
#define N_UNROLL 8

/*
 * Compute C += A*B for some really tiny subblocks of A, B, and C
 */
template <typename MatrixA, typename MatrixB, typename MatrixC>
void my_dgemm_micro_kernel(int k, const MatrixA& A, const MatrixB& B, MatrixC& C)
{
    const double* A_ptr = A.data();
    const double* B_ptr = B.data();

    int lda = A.outerStride();
    int ldb = B.outerStride();

    __m256d C_0_0123 = _mm256_setzero_pd();
    __m256d C_0_4567 = _mm256_setzero_pd();
    __m256d C_1_0123 = _mm256_setzero_pd();
    __m256d C_1_4567 = _mm256_setzero_pd();
    __m256d C_2_0123 = _mm256_setzero_pd();
    __m256d C_2_4567 = _mm256_setzero_pd();
    __m256d C_3_0123 = _mm256_setzero_pd();
    __m256d C_3_4567 = _mm256_setzero_pd();
    __m256d C_4_0123 = _mm256_setzero_pd();
    __m256d C_4_4567 = _mm256_setzero_pd();
    __m256d C_5_0123 = _mm256_setzero_pd();
    __m256d C_5_4567 = _mm256_setzero_pd();

    for (int p = 0;p < k;p++)
    {
        __m256d B_0123 = _mm256_loadu_pd(B_ptr + 0);
        __m256d B_4567 = _mm256_loadu_pd(B_ptr + 4);

        __m256d A_0 = _mm256_broadcast_sd(A_ptr + 0*lda);
        __m256d A_1 = _mm256_broadcast_sd(A_ptr + 1*lda);
        C_0_0123 = _mm256_add_pd(C_0_0123, _mm256_mul_pd(A_0, B_0123));
        C_0_4567 = _mm256_add_pd(C_0_4567, _mm256_mul_pd(A_0, B_4567));
        C_1_0123 = _mm256_add_pd(C_1_0123, _mm256_mul_pd(A_1, B_0123));
        C_1_4567 = _mm256_add_pd(C_1_4567, _mm256_mul_pd(A_1, B_4567));

        __m256d A_2 = _mm256_broadcast_sd(A_ptr + 2*lda);
        __m256d A_3 = _mm256_broadcast_sd(A_ptr + 3*lda);
        C_2_0123 = _mm256_add_pd(C_2_0123, _mm256_mul_pd(A_2, B_0123));
        C_2_4567 = _mm256_add_pd(C_2_4567, _mm256_mul_pd(A_2, B_4567));
        C_3_0123 = _mm256_add_pd(C_3_0123, _mm256_mul_pd(A_3, B_0123));
        C_3_4567 = _mm256_add_pd(C_3_4567, _mm256_mul_pd(A_3, B_4567));

        __m256d A_4 = _mm256_broadcast_sd(A_ptr + 4*lda);
        __m256d A_5 = _mm256_broadcast_sd(A_ptr + 5*lda);
        C_4_0123 = _mm256_add_pd(C_4_0123, _mm256_mul_pd(A_4, B_0123));
        C_4_4567 = _mm256_add_pd(C_4_4567, _mm256_mul_pd(A_4, B_4567));
        C_5_0123 = _mm256_add_pd(C_5_0123, _mm256_mul_pd(A_5, B_0123));
        C_5_4567 = _mm256_add_pd(C_5_4567, _mm256_mul_pd(A_5, B_4567));

        A_ptr++;
        B_ptr += ldb;
    }

    C_0_0123 = _mm256_add_pd(C_0_0123, _mm256_loadu_pd(&C(0,0)));
    C_0_4567 = _mm256_add_pd(C_0_4567, _mm256_loadu_pd(&C(0,4)));
    C_1_0123 = _mm256_add_pd(C_1_0123, _mm256_loadu_pd(&C(1,0)));
    C_1_4567 = _mm256_add_pd(C_1_4567, _mm256_loadu_pd(&C(1,4)));
    C_2_0123 = _mm256_add_pd(C_2_0123, _mm256_loadu_pd(&C(2,0)));
    C_2_4567 = _mm256_add_pd(C_2_4567, _mm256_loadu_pd(&C(2,4)));
    C_3_0123 = _mm256_add_pd(C_3_0123, _mm256_loadu_pd(&C(3,0)));
    C_3_4567 = _mm256_add_pd(C_3_4567, _mm256_loadu_pd(&C(3,4)));
    C_4_0123 = _mm256_add_pd(C_4_0123, _mm256_loadu_pd(&C(4,0)));
    C_4_4567 = _mm256_add_pd(C_4_4567, _mm256_loadu_pd(&C(4,4)));
    C_5_0123 = _mm256_add_pd(C_5_0123, _mm256_loadu_pd(&C(5,0)));
    C_5_4567 = _mm256_add_pd(C_5_4567, _mm256_loadu_pd(&C(5,4)));

    _mm256_storeu_pd(&C(0,0), C_0_0123);
    _mm256_storeu_pd(&C(0,4), C_0_4567);
    _mm256_storeu_pd(&C(1,0), C_1_0123);
    _mm256_storeu_pd(&C(1,4), C_1_4567);
    _mm256_storeu_pd(&C(2,0), C_2_0123);
    _mm256_storeu_pd(&C(2,4), C_2_4567);
    _mm256_storeu_pd(&C(3,0), C_3_0123);
    _mm256_storeu_pd(&C(3,4), C_3_4567);
    _mm256_storeu_pd(&C(4,0), C_4_0123);
    _mm256_storeu_pd(&C(4,4), C_4_4567);
    _mm256_storeu_pd(&C(5,0), C_5_0123);
    _mm256_storeu_pd(&C(5,4), C_5_4567);
}

/*
 * Compute C += A*B for some subblocks of A, B, and C
 */
template <typename MatrixA, typename MatrixB, typename MatrixC>
void my_dgemm_inner_kernel(int m, int n, int k,
                           const MatrixA& A, const MatrixB& B, MatrixC& C)
{
    for (int j = 0;j < n;j += N_UNROLL)
    {
        for (int i = 0;i < m;i += M_UNROLL)
        {
            auto A_sub = A.block(i, 0, M_UNROLL,        k);
            auto B_sub = B.block(0, j,        k, N_UNROLL);
            auto C_sub = C.block(i, j, M_UNROLL, N_UNROLL);

            my_dgemm_micro_kernel(k, A_sub, B_sub, C_sub);
        }
    }
}

/*
 * Compute C += A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
    /*
     * Step 5:
     *
     * Performance looks good for small matrices, but what about big ones?
     * Even though the GEMM operation reuses data, we have to make sure that
     * it stays in the caches long enough to get used enough times. The main
     * optimization to provide this temporal locality is blocking.
     *
     * Like with unrolling, we will make two copied of each relevant loop.
     * However, we will not fully expand the inner loops (generally we won't
     * know how many iterations they will do because of remainders). When the
     * loops are blocked, we will now come back to the same data sooner (with
     * a frequency controlled by the block sizes). Another way to think of this
     * is that we've limited the amount of data we're using in the inner kernel
     * so that it fits in the caches.
     *
     * Again, we've put the inner loops into a kernel. Now we have an inner
     * kernel (see K. Goto and R.A. van de Geijn, TOMS, 34, Article 12
     * (2008) for why it is called this), and a micro kernel (see F.G. Van Zee
     * and R.A. van de Geijn, TOMS, 41, Article 14 (2015)).
     *
     * Exercise: based on the size of M_BLOCK*K_BLOCK, N_UNROLL*K_BLOCK, etc.,
     * can you guess what pieces of data are kept in each level of cache?
     * (Hint: L1 is 32KB, L2 is 256KB, and L3 is large (>1MB per core))
     */
    for (int j = 0;j < n;j += N_BLOCK)
    {
        int n_sub = std::min(N_BLOCK, n-j);

        for (int p = 0;p < k;p += K_BLOCK)
        {
            int k_sub = std::min(K_BLOCK, k-p);

            auto B_sub = B.block(p, j, k_sub, n_sub);

            for (int i = 0;i < m;i += M_BLOCK)
            {
                int m_sub = std::min(M_BLOCK, m-i);

                auto A_sub = A.block(i, p, m_sub, k_sub);
                auto C_sub = C.block(i, j, m_sub, n_sub);

                my_dgemm_inner_kernel(m_sub, n_sub, k_sub, A_sub, B_sub, C_sub);
            }
        }
    }
}
