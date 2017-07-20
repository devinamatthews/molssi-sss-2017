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
template <typename MatrixC>
void my_dgemm_micro_kernel(int k, const double* A, const double* B, MatrixC& C)
{
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
        __m256d B_0123 = _mm256_loadu_pd(B + 0);
        __m256d B_4567 = _mm256_loadu_pd(B + 4);

        __m256d A_0 = _mm256_broadcast_sd(A + 0);
        __m256d A_1 = _mm256_broadcast_sd(A + 1);
        C_0_0123 = _mm256_add_pd(C_0_0123, _mm256_mul_pd(A_0, B_0123));
        C_0_4567 = _mm256_add_pd(C_0_4567, _mm256_mul_pd(A_0, B_4567));
        C_1_0123 = _mm256_add_pd(C_1_0123, _mm256_mul_pd(A_1, B_0123));
        C_1_4567 = _mm256_add_pd(C_1_4567, _mm256_mul_pd(A_1, B_4567));

        __m256d A_2 = _mm256_broadcast_sd(A + 2);
        __m256d A_3 = _mm256_broadcast_sd(A + 3);
        C_2_0123 = _mm256_add_pd(C_2_0123, _mm256_mul_pd(A_2, B_0123));
        C_2_4567 = _mm256_add_pd(C_2_4567, _mm256_mul_pd(A_2, B_4567));
        C_3_0123 = _mm256_add_pd(C_3_0123, _mm256_mul_pd(A_3, B_0123));
        C_3_4567 = _mm256_add_pd(C_3_4567, _mm256_mul_pd(A_3, B_4567));

        __m256d A_4 = _mm256_broadcast_sd(A + 4);
        __m256d A_5 = _mm256_broadcast_sd(A + 5);
        C_4_0123 = _mm256_add_pd(C_4_0123, _mm256_mul_pd(A_4, B_0123));
        C_4_4567 = _mm256_add_pd(C_4_4567, _mm256_mul_pd(A_4, B_4567));
        C_5_0123 = _mm256_add_pd(C_5_0123, _mm256_mul_pd(A_5, B_0123));
        C_5_4567 = _mm256_add_pd(C_5_4567, _mm256_mul_pd(A_5, B_4567));

        A += M_UNROLL;
        B += N_UNROLL;
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
template <typename MatrixC>
void my_dgemm_inner_kernel(int m, int n, int k,
                           const double* A, const double* B, MatrixC& C)
{
    /*
     * NB: we apply the strength reduction optimization to the indexing of
     * A and B here. However, every time we go through the j loop, we want to
     * start with a fresh pointer to the beginning of A. We accomplish this by
     * "undoing" the updates to A at the end of the loop over i.
     *
     * Exercise: how else could you fix A each iteration of the j loop?
     */
    for (int j = 0;j < n;j += N_UNROLL)
    {
        for (int i = 0;i < m;i += M_UNROLL)
        {
            auto C_sub = C.block(i, j, M_UNROLL, N_UNROLL);

            my_dgemm_micro_kernel(k, A, B, C_sub);

            A += M_UNROLL*k;
        }

        A -= m*k;
        B += N_UNROLL*k;
    }
}

/*
 * Pack a panel of A into contiguous memory
 */
template <typename MatrixA>
void my_dgemm_pack_a(int m, int k, const MatrixA& A, double* A_pack)
{
    const double* A_ptr = A.data();

    int lda = A.outerStride();

    /*
     * NB: we apply the strength reduction optimization to the indexing of
     * A here as well. Note that the way we update A_ptr in the loop over p
     * conflicts with how we update in the loop over i. To fix this we "undo"
     * the updates from the p loop in the latter update.
     */
    for (int i = 0;i < m;i += M_UNROLL)
    {
        for (int p = 0;p < k;p++)
        {
            A_pack[0] = A_ptr[0*lda];
            A_pack[1] = A_ptr[1*lda];
            A_pack[2] = A_ptr[2*lda];
            A_pack[3] = A_ptr[3*lda];
            A_pack[4] = A_ptr[4*lda];
            A_pack[5] = A_ptr[5*lda];

            A_pack += M_UNROLL;
            A_ptr++;
        }

        A_ptr += M_UNROLL*lda - k;
    }
}

/*
 * Pack a panel of B into contiguous memory
 */
template <typename MatrixB>
void my_dgemm_pack_b(int n, int k, const MatrixB& B, double* B_pack)
{
    const double* B_ptr = B.data();

    int ldb = B.outerStride();

    /*
     * NB: we apply the strength reduction optimization to the indexing of
     * B here as well. Note that the way we update B_ptr in the loop over p
     * conflicts with how we update in the loop over j. To fix this we "undo"
     * the updates from the p loop in the latter update.
     */
    for (int j = 0;j < n;j += N_UNROLL)
    {
        for (int p = 0;p < k;p++)
        {
            B_pack[0] = B_ptr[0];
            B_pack[1] = B_ptr[1];
            B_pack[2] = B_ptr[2];
            B_pack[3] = B_ptr[3];
            B_pack[4] = B_ptr[4];
            B_pack[5] = B_ptr[5];
            B_pack[6] = B_ptr[6];
            B_pack[7] = B_ptr[7];

            B_pack += N_UNROLL;
            B_ptr += ldb;
        }

        B_ptr += N_UNROLL - ldb*k;
    }
}

/*
 * NB: we put the space for our packed pieces of A and B here as global
 * variables.
 *
 * Exercise: what are some reasons why we would want to put them here and not
 * as variables in my_dgemm?
 */
static double A_pack[M_BLOCK*K_BLOCK];
static double B_pack[N_BLOCK*K_BLOCK];

/*
 * Compute C += A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
    /*
     * Step 6:
     *
     * Even though we've blocked the loops, we may still have extra data
     * movement. This is because the blocks of A and B that we are accessing
     * may be spread over a large area of memory (when they are pieces of a
     * much larger matrix).
     *
     * In order to optimize how this data is used, we pack pieces of A and B
     * into buffers, where the data is stored contiguously and in exactly the
     * same order in which it will be accessed in the kernel.
     *
     * See additional notes in the packing functions my_dgemm_pack_[ab].
     *
     * Exercise: in Step 2 we ignored the fact that e.g. m%M_UNROLL may not
     * be zero. How does this affect the packing functions? How would the
     * changes in the packing function then affect the kernels (the micro-
     * kernel specifically)?
     */
    for (int j = 0;j < n;j += N_BLOCK)
    {
        int n_sub = std::min(N_BLOCK, n-j);

        for (int p = 0;p < k;p += K_BLOCK)
        {
            int k_sub = std::min(K_BLOCK, k-p);

            auto B_sub = B.block(p, j, k_sub, n_sub);

            my_dgemm_pack_b(n_sub, k_sub, B_sub, B_pack);

            for (int i = 0;i < m;i += M_BLOCK)
            {
                int m_sub = std::min(M_BLOCK, m-i);

                auto A_sub = A.block(i, p, m_sub, k_sub);
                auto C_sub = C.block(i, j, m_sub, n_sub);

                my_dgemm_pack_a(m_sub, k_sub, A_sub, A_pack);

                my_dgemm_inner_kernel(m_sub, n_sub, k_sub, A_pack, B_pack, C_sub);
            }
        }
    }
}
