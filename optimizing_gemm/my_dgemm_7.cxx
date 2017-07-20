#include "common.hpp"

#include "immintrin.h"

#define M_BLOCK 72
#define N_BLOCK 4080
#define K_BLOCK 256

#define M_UNROLL 6
#define N_UNROLL 8
#define K_UNROLL 4

/*
 * Compute C += A*B for some really tiny subblocks of A, B, and C
 */
template <typename MatrixC>
void my_dgemm_micro_kernel(int k, const double* A, const double* B, MatrixC& C)
{
    /*
     * Step 7:
     *
     * Now performance is pretty good, but still not quite to the level of
     * a professional BLAS implementation. Some additional optimizations that
     * can help at this point are:
     *
     * 1) More unrolling (of the loop over p this time): each time we go through
     *    this loop, we have to update the values of A, B, and k, and then do
     *    a branch. This can cause pipeline stalls in the floating point units
     *    and lower performance. Unrolling reduces the frequency of these
     *    updates and branches.
     *
     * 2) Prefetching: because of the order in which we access data, the values
     *    of B that we use are (almost) always already going to be in cache.
     *    But, the values of A may have already been pushed out of cache (that
     *    is, the L1 cache---they are probably still in the L2 cache).
     *    Additionally, the values of C that we're updating are probably not
     *    in any level of cache at all. By explicitly prefetching these ahead
     *    of time we avoid delays.
     */

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

    _mm_prefetch(&C(0,0), _MM_HINT_T0);
    _mm_prefetch(&C(1,0), _MM_HINT_T0);
    _mm_prefetch(&C(2,0), _MM_HINT_T0);
    _mm_prefetch(&C(3,0), _MM_HINT_T0);
    _mm_prefetch(&C(4,0), _MM_HINT_T0);
    _mm_prefetch(&C(5,0), _MM_HINT_T0);

    __m256d B_0123, B_4567, A_0, A_1, A_2, A_3, A_4, A_5;

    #define NTH_ITERATION(n) \
    \
    B_0123 = _mm256_loadu_pd(B + 0 + N_UNROLL*n); \
    B_4567 = _mm256_loadu_pd(B + 4 + N_UNROLL*n); \
    \
    A_0 = _mm256_broadcast_sd(A + 0 + M_UNROLL*n); \
    A_1 = _mm256_broadcast_sd(A + 1 + M_UNROLL*n); \
    C_0_0123 = _mm256_add_pd(C_0_0123, _mm256_mul_pd(A_0, B_0123)); \
    C_0_4567 = _mm256_add_pd(C_0_4567, _mm256_mul_pd(A_0, B_4567)); \
    C_1_0123 = _mm256_add_pd(C_1_0123, _mm256_mul_pd(A_1, B_0123)); \
    C_1_4567 = _mm256_add_pd(C_1_4567, _mm256_mul_pd(A_1, B_4567)); \
    \
    A_2 = _mm256_broadcast_sd(A + 2 + M_UNROLL*n); \
    A_3 = _mm256_broadcast_sd(A + 3 + M_UNROLL*n); \
    C_2_0123 = _mm256_add_pd(C_2_0123, _mm256_mul_pd(A_2, B_0123)); \
    C_2_4567 = _mm256_add_pd(C_2_4567, _mm256_mul_pd(A_2, B_4567)); \
    C_3_0123 = _mm256_add_pd(C_3_0123, _mm256_mul_pd(A_3, B_0123)); \
    C_3_4567 = _mm256_add_pd(C_3_4567, _mm256_mul_pd(A_3, B_4567)); \
    \
    A_4 = _mm256_broadcast_sd(A + 4 + M_UNROLL*n); \
    A_5 = _mm256_broadcast_sd(A + 5 + M_UNROLL*n); \
    C_4_0123 = _mm256_add_pd(C_4_0123, _mm256_mul_pd(A_4, B_0123)); \
    C_4_4567 = _mm256_add_pd(C_4_4567, _mm256_mul_pd(A_4, B_4567)); \
    C_5_0123 = _mm256_add_pd(C_5_0123, _mm256_mul_pd(A_5, B_0123)); \
    C_5_4567 = _mm256_add_pd(C_5_4567, _mm256_mul_pd(A_5, B_4567));

    for (;k > 0;k -= K_UNROLL)
    {
        _mm_prefetch(A + 64, _MM_HINT_T0);
        NTH_ITERATION(0);

        _mm_prefetch(A + 72, _MM_HINT_T0);
        NTH_ITERATION(1);

        _mm_prefetch(A + 80, _MM_HINT_T0);
        NTH_ITERATION(2);

        NTH_ITERATION(3);

        A += M_UNROLL*K_UNROLL;
        B += N_UNROLL*K_UNROLL;
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

static double A_pack[M_BLOCK*K_BLOCK];
static double B_pack[N_BLOCK*K_BLOCK];

/*
 * Compute C += A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
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
