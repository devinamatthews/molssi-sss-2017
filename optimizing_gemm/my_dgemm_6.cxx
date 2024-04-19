#include "common.hpp"

#define M_UNROLL 6
#define N_UNROLL 8

#if defined(__x86_64__)

#define M_BLOCK 72
#define N_BLOCK 4080
#define K_BLOCK 256

#include "immintrin.h"

/*
 * Compute C += A*B for some subblocks of A, B, and C
 */
template <typename MatrixC>
void my_dgemm_micro_kernel(int k, const double* A, const double* B, MatrixC& C)
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
    const double* A_ptr = A;
    const double* B_ptr = B;

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

        __m256d A_0 = _mm256_broadcast_sd(A_ptr + 0);
        __m256d A_1 = _mm256_broadcast_sd(A_ptr + 1);
        C_0_0123 = _mm256_add_pd(C_0_0123, _mm256_mul_pd(A_0, B_0123));
        C_0_4567 = _mm256_add_pd(C_0_4567, _mm256_mul_pd(A_0, B_4567));
        C_1_0123 = _mm256_add_pd(C_1_0123, _mm256_mul_pd(A_1, B_0123));
        C_1_4567 = _mm256_add_pd(C_1_4567, _mm256_mul_pd(A_1, B_4567));

        __m256d A_2 = _mm256_broadcast_sd(A_ptr + 2);
        __m256d A_3 = _mm256_broadcast_sd(A_ptr + 3);
        C_2_0123 = _mm256_add_pd(C_2_0123, _mm256_mul_pd(A_2, B_0123));
        C_2_4567 = _mm256_add_pd(C_2_4567, _mm256_mul_pd(A_2, B_4567));
        C_3_0123 = _mm256_add_pd(C_3_0123, _mm256_mul_pd(A_3, B_0123));
        C_3_4567 = _mm256_add_pd(C_3_4567, _mm256_mul_pd(A_3, B_4567));

        __m256d A_4 = _mm256_broadcast_sd(A_ptr + 4);
        __m256d A_5 = _mm256_broadcast_sd(A_ptr + 5);
        C_4_0123 = _mm256_add_pd(C_4_0123, _mm256_mul_pd(A_4, B_0123));
        C_4_4567 = _mm256_add_pd(C_4_4567, _mm256_mul_pd(A_4, B_4567));
        C_5_0123 = _mm256_add_pd(C_5_0123, _mm256_mul_pd(A_5, B_0123));
        C_5_4567 = _mm256_add_pd(C_5_4567, _mm256_mul_pd(A_5, B_4567));

        A_ptr += M_UNROLL;
        B_ptr += N_UNROLL;
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

#define M_BLOCK 240
#define N_BLOCK 8184
#define K_BLOCK 256

#include "arm_neon.h"

/*
 * Compute C += A*B for some subblocks of A, B, and C
 */
template <typename MatrixC>
void my_dgemm_micro_kernel(int k, const double* A, const double* B, MatrixC& C)
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
    const double* __restrict__ A_ptr = A;
    const double* __restrict__ B_ptr = B;

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

        float64x2_t A_0 = vld1q_dup_f64(A_ptr + 0);
        float64x2_t A_1 = vld1q_dup_f64(A_ptr + 1);
        float64x2_t A_2 = vld1q_dup_f64(A_ptr + 2);
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

        float64x2_t A_3 = vld1q_dup_f64(A_ptr + 3);
        float64x2_t A_4 = vld1q_dup_f64(A_ptr + 4);
        float64x2_t A_5 = vld1q_dup_f64(A_ptr + 5);
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

        A_ptr += M_UNROLL;
        B_ptr += N_UNROLL;
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
