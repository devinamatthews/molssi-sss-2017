#include "common.hpp"

#include "immintrin.h"

#define M_UNROLL 6
#define N_UNROLL 8

/*
 * Compute C += A*B for some subblocks of A, B, and C
 */
template <typename MatrixA, typename MatrixB, typename MatrixC>
void my_dgemm_kernel(int k, const MatrixA& A, const MatrixB& B, MatrixC& C)
{
    /*
     * Step 4:
     *
     * Unfortunately, even with all the help we've given it, the compiler still
     * can't apply the most important optimization to this code: vectorization.
     * Note: if you're using the Intel compiler it may actually have done a
     * good job... So, we are forced to do the vectorization manually.
     *
     * The code is a direct translation of Step 3, except that:
     *
     * 1) Sets of four contiguous elements (of B and C) are now loaded and
     *    operated on as vectors.
     *
     * 2) Elements of A are broadcast (i.e. the vector where they are loaded
     *    has the same value in all four slots).
     *
     * 3) The compiler doesn't necessarily use exactly the instructions that
     *    we tell it to or put them in the same order. For example,
     *
     *    _mm256_add_pd(..., _mm256_mul_pd(...))
     *
     *    may be replaced by the compiler with a single FMA (fused multiply-add)
     *    instruction.
     *
     *  https://software.intel.com/sites/landingpage/IntrinsicsGuide and
     *  https://www.intel.com/content/dam/www/public/us/en/documents/manuals/
     *      64-ia-32-architectures-software-developer-instruction-set-reference-manual-325383.pdf
     *  are good sources on x86-64 intrinsics and instructions.
     *
     * Exercise: this kernel uses the broadcast instruction. If we re-ordered
     * the updates to C(i,j), what other kinds of instructions could we use
     * to build this kernel?
     */
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
 * Compute C += A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
    for (int j = 0;j < n;j += N_UNROLL)
    {
        for (int i = 0;i < m;i += M_UNROLL)
        {
            auto A_sub = A.block(i, 0, M_UNROLL,        k);
            auto B_sub = B.block(0, j,        k, N_UNROLL);
            auto C_sub = C.block(i, j, M_UNROLL, N_UNROLL);

            my_dgemm_kernel(k, A_sub, B_sub, C_sub);
        }
    }
}
