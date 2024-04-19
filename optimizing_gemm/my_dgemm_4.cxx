#include "common.hpp"

#define M_UNROLL 6
#define N_UNROLL 8

#if defined(__x86_64__)

#include "immintrin.h"

/*
 * Compute C += A*B for some subblocks of A, B, and C
 */
template <typename MatrixA, typename MatrixB, typename MatrixC>
void my_dgemm_micro_kernel(int k, const MatrixA& A, const MatrixB& B, MatrixC& C)
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

#elif defined(__aarch64__)

#include "arm_neon.h"

/*
 * Compute C += A*B for some subblocks of A, B, and C
 */
template <typename MatrixA, typename MatrixB, typename MatrixC>
void my_dgemm_micro_kernel(int k, const MatrixA& A, const MatrixB& B, MatrixC& C)
{
    /*
     * Step 4:
     *
     * Unfortunately, even with all the help we've given it, the compiler still
     * can't apply all important optimization to this code.
     * Note: if you're using the Intel compiler it may actually have done a
     * good job... So, we are forced to do some optimizations manually. In
     * this case we will use assembly intrinsics to very explicitly tell the
     * compiler what we want.
     *
     * The code is a direct translation of Step 3, except that:
     *
     * 1) Sets of four contiguous elements (of B and C) are now loaded and
     *    operated on as vectors.
     *
     * 2) Elements of A are broadcast (i.e. the vector where they are loaded
     *    has the same value in both slots).
     *
     * 3) We will combine the * and + operations into a single FMA (fused multiply-add)
     *    instruction.
     *
     *  https://developer.arm.com/architectures/instruction-sets/intrinsics/ and
     *  https://developer.arm.com/documentation/ddi0596/2021-03?lang=en
     *  are good sources on ARM AArch64 intrinsics and instructions.
     *
     * Exercise: this kernel uses the broadcast instruction. If we re-ordered
     * the updates to C(i,j), what other kinds of instructions could we use
     * to build this kernel?
     */
    const double* __restrict__ A_ptr = A.data();
    const double* __restrict__ B_ptr = B.data();

    /*
     * NB: &A(i,p) = A.data() + i*lda + p and similarly for B
     */
    int lda = A.outerStride();
    int ldb = B.outerStride();

    float64x2_t C_0_01 = vdupq_n_f64(0.0);
    float64x2_t C_0_23 = vdupq_n_f64(0.0);
    float64x2_t C_0_45 = vdupq_n_f64(0.0);
    float64x2_t C_0_67 = vdupq_n_f64(0.0);
    float64x2_t C_1_01 = vdupq_n_f64(0.0);
    float64x2_t C_1_23 = vdupq_n_f64(0.0);
    float64x2_t C_1_45 = vdupq_n_f64(0.0);
    float64x2_t C_1_67 = vdupq_n_f64(0.0);
    float64x2_t C_2_01 = vdupq_n_f64(0.0);
    float64x2_t C_2_23 = vdupq_n_f64(0.0);
    float64x2_t C_2_45 = vdupq_n_f64(0.0);
    float64x2_t C_2_67 = vdupq_n_f64(0.0);
    float64x2_t C_3_01 = vdupq_n_f64(0.0);
    float64x2_t C_3_23 = vdupq_n_f64(0.0);
    float64x2_t C_3_45 = vdupq_n_f64(0.0);
    float64x2_t C_3_67 = vdupq_n_f64(0.0);
    float64x2_t C_4_01 = vdupq_n_f64(0.0);
    float64x2_t C_4_23 = vdupq_n_f64(0.0);
    float64x2_t C_4_45 = vdupq_n_f64(0.0);
    float64x2_t C_4_67 = vdupq_n_f64(0.0);
    float64x2_t C_5_01 = vdupq_n_f64(0.0);
    float64x2_t C_5_23 = vdupq_n_f64(0.0);
    float64x2_t C_5_45 = vdupq_n_f64(0.0);
    float64x2_t C_5_67 = vdupq_n_f64(0.0);

    for (int p = 0;p < k;p++)
    {
        float64x2_t B_01 = vld1q_f64(B_ptr + 0);
        float64x2_t B_23 = vld1q_f64(B_ptr + 2);
        float64x2_t B_45 = vld1q_f64(B_ptr + 4);
        float64x2_t B_67 = vld1q_f64(B_ptr + 6);

        float64x2_t A_0 = vld1q_dup_f64(A_ptr + 0*lda);
        float64x2_t A_1 = vld1q_dup_f64(A_ptr + 1*lda);
        float64x2_t A_2 = vld1q_dup_f64(A_ptr + 2*lda);
        C_0_01 = vfmaq_f64(C_0_01, A_0, B_01);
        C_0_23 = vfmaq_f64(C_0_23, A_0, B_23);
        C_0_45 = vfmaq_f64(C_0_45, A_0, B_45);
        C_0_67 = vfmaq_f64(C_0_67, A_0, B_67);
        C_1_01 = vfmaq_f64(C_1_01, A_1, B_01);
        C_1_23 = vfmaq_f64(C_1_23, A_1, B_23);
        C_1_45 = vfmaq_f64(C_1_45, A_1, B_45);
        C_1_67 = vfmaq_f64(C_1_67, A_1, B_67);
        C_2_01 = vfmaq_f64(C_2_01, A_2, B_01);
        C_2_23 = vfmaq_f64(C_2_23, A_2, B_23);
        C_2_45 = vfmaq_f64(C_2_45, A_2, B_45);
        C_2_67 = vfmaq_f64(C_2_67, A_2, B_67);

        float64x2_t A_3 = vld1q_dup_f64(A_ptr + 3*lda);
        float64x2_t A_4 = vld1q_dup_f64(A_ptr + 4*lda);
        float64x2_t A_5 = vld1q_dup_f64(A_ptr + 5*lda);
        C_3_01 = vfmaq_f64(C_3_01, A_3, B_01);
        C_3_23 = vfmaq_f64(C_3_23, A_3, B_23);
        C_3_45 = vfmaq_f64(C_3_45, A_3, B_45);
        C_3_67 = vfmaq_f64(C_3_67, A_3, B_67);
        C_4_01 = vfmaq_f64(C_4_01, A_4, B_01);
        C_4_23 = vfmaq_f64(C_4_23, A_4, B_23);
        C_4_45 = vfmaq_f64(C_4_45, A_4, B_45);
        C_4_67 = vfmaq_f64(C_4_67, A_4, B_67);
        C_5_01 = vfmaq_f64(C_5_01, A_5, B_01);
        C_5_23 = vfmaq_f64(C_5_23, A_5, B_23);
        C_5_45 = vfmaq_f64(C_5_45, A_5, B_45);
        C_5_67 = vfmaq_f64(C_5_67, A_5, B_67);

        A_ptr++;
        B_ptr += ldb;
    }

    C_0_01 = vaddq_f64(C_0_01, vld1q_f64(&C(0,0)));
    C_0_23 = vaddq_f64(C_0_23, vld1q_f64(&C(0,2)));
    C_0_45 = vaddq_f64(C_0_45, vld1q_f64(&C(0,4)));
    C_0_67 = vaddq_f64(C_0_67, vld1q_f64(&C(0,6)));
    C_1_01 = vaddq_f64(C_1_01, vld1q_f64(&C(1,0)));
    C_1_23 = vaddq_f64(C_1_23, vld1q_f64(&C(1,2)));
    C_1_45 = vaddq_f64(C_1_45, vld1q_f64(&C(1,4)));
    C_1_67 = vaddq_f64(C_1_67, vld1q_f64(&C(1,6)));
    C_2_01 = vaddq_f64(C_2_01, vld1q_f64(&C(2,0)));
    C_2_23 = vaddq_f64(C_2_23, vld1q_f64(&C(2,2)));
    C_2_45 = vaddq_f64(C_2_45, vld1q_f64(&C(2,4)));
    C_2_67 = vaddq_f64(C_2_67, vld1q_f64(&C(2,6)));
    C_3_01 = vaddq_f64(C_3_01, vld1q_f64(&C(3,0)));
    C_3_23 = vaddq_f64(C_3_23, vld1q_f64(&C(3,2)));
    C_3_45 = vaddq_f64(C_3_45, vld1q_f64(&C(3,4)));
    C_3_67 = vaddq_f64(C_3_67, vld1q_f64(&C(3,6)));
    C_4_01 = vaddq_f64(C_4_01, vld1q_f64(&C(4,0)));
    C_4_23 = vaddq_f64(C_4_23, vld1q_f64(&C(4,2)));
    C_4_45 = vaddq_f64(C_4_45, vld1q_f64(&C(4,4)));
    C_4_67 = vaddq_f64(C_4_67, vld1q_f64(&C(4,6)));
    C_5_01 = vaddq_f64(C_5_01, vld1q_f64(&C(5,0)));
    C_5_23 = vaddq_f64(C_5_23, vld1q_f64(&C(5,2)));
    C_5_45 = vaddq_f64(C_5_45, vld1q_f64(&C(5,4)));
    C_5_67 = vaddq_f64(C_5_67, vld1q_f64(&C(5,6)));

    vst1q_f64(&C(0,0), C_0_01);
    vst1q_f64(&C(0,2), C_0_23);
    vst1q_f64(&C(0,4), C_0_45);
    vst1q_f64(&C(0,6), C_0_67);
    vst1q_f64(&C(1,0), C_1_01);
    vst1q_f64(&C(1,2), C_1_23);
    vst1q_f64(&C(1,4), C_1_45);
    vst1q_f64(&C(1,6), C_1_67);
    vst1q_f64(&C(2,0), C_2_01);
    vst1q_f64(&C(2,2), C_2_23);
    vst1q_f64(&C(2,4), C_2_45);
    vst1q_f64(&C(2,6), C_2_67);
    vst1q_f64(&C(3,0), C_3_01);
    vst1q_f64(&C(3,2), C_3_23);
    vst1q_f64(&C(3,4), C_3_45);
    vst1q_f64(&C(3,6), C_3_67);
    vst1q_f64(&C(4,0), C_4_01);
    vst1q_f64(&C(4,2), C_4_23);
    vst1q_f64(&C(4,4), C_4_45);
    vst1q_f64(&C(4,6), C_4_67);
    vst1q_f64(&C(5,0), C_5_01);
    vst1q_f64(&C(5,2), C_5_23);
    vst1q_f64(&C(5,4), C_5_45);
    vst1q_f64(&C(5,6), C_5_67);
}

#else

#error "Unknown architecture"

#endif

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

            my_dgemm_micro_kernel(k, A_sub, B_sub, C_sub);
        }
    }
}
