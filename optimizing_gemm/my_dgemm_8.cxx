#include "common.hpp"

#include "immintrin.h"
#include <stdint.h>

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
void my_dgemm_micro_kernel(int64_t k, const double* A, const double* B, MatrixC& C)
{
    /*
     * Step 8:
     *
     * What happened? Step 7 was supposed to make things faster, but instead
     * it is slower?! (Note, if you are using the Intel compiler it may have
     * done a better job than gcc) The reason is because the compiler is trying
     * to generate the best code it knows how, but it can only optimize for the
     * general case. HPC code (like GEMM) is not the usual case, and the
     * compiler can sometimes mess up even code with intrinsics badly.
     *
     * Instead, we will simply translate the intrincis in Step 7 to inline
     * assembly. Luckily, you don't have to optimize this far very often.
     *
     * The performance is *still* not quite up to BLAS, though. The last ~10-20%
     * of relative speed takes ~80-90% of the effort, so unless you really need
     * that last little bit, know when to call it quits.
     *
     * Some thing that MKL or OpenBLAS does that we don't:
     *
     * 1) Writing the whole inner kernel (with the micro-kernel inlined) in
     *    assembly.
     *
     * 2) More prefetching, including to higher levels of caches.
     *
     * 3) Dynamic adjustment of the blocksizes.
     *
     * 4) Optimizing the packing functions.
     *
     * 5) Other forms of Black Magic.
     *
     * Exercise: why do k and ldc have to be int64_t instead of just int?
     */

    double* C_ptr = C.data();
    int64_t ldc = C.outerStride();

    __asm__ volatile
    (
    "vzeroall                                    \n\t" // zero all ymm registers
    "                                            \n\t"
    "movq              %[k], %%rsi               \n\t" // load k
    "sarq                $2, %%rsi               \n\t" // divide k by K_UNROLL
    "                                            \n\t"
    "movq              %[a], %%rax               \n\t" // load address of a
    "movq              %[b], %%rbx               \n\t" // load address of b
    "                                            \n\t"
    "movq              %[c], %%rcx               \n\t" // load address of c
    "movq            %[ldc], %%rdi               \n\t" // load ldc
    "leaq        (,%%rdi,8), %%rdi               \n\t" // ldc *= sizeof(double)
    "                                            \n\t"
    "leaq   (%%rdi,%%rdi,2), %%r13               \n\t" // r13 = 3*ldc
    "leaq   (%%rcx,%%r13,1), %%rdx               \n\t" // rdx = c + 3*ldc
    "                                            \n\t" // (i.e. &c(3,0))
    "                                            \n\t"
    "prefetcht0        (%%rcx)                   \n\t" // prefetch c(0,0)
    "prefetcht0        (%%rcx,%%rdi)             \n\t" // prefetch c(1,0)
    "prefetcht0        (%%rcx,%%rdi,2)           \n\t" // prefetch c(2,0)
    "prefetcht0        (%%rdx)                   \n\t" // prefetch c(3,0)
    "prefetcht0        (%%rdx,%%rdi)             \n\t" // prefetch c(4,0)
    "prefetcht0        (%%rdx,%%rdi,2)           \n\t" // prefetch c(5,0)
    "                                            \n\t"
    ".DLOOPKITER:                                \n\t"
    "                                            \n\t"
    "prefetcht0        64 * 8(%%rax)             \n\t"
    "                                            \n\t"
    "vmovapd            0 * 32(%%rbx), %%ymm0    \n\t" // iteration 0
    "vmovapd            1 * 32(%%rbx), %%ymm1    \n\t"
    "                                            \n\t"
    "vbroadcastsd       0 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       1 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd       2 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       3 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd       4 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       5 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "prefetcht0        72 * 8(%%rax)             \n\t"
    "                                            \n\t"
    "vmovapd            2 * 32(%%rbx), %%ymm0    \n\t" // iteration 1
    "vmovapd            3 * 32(%%rbx), %%ymm1    \n\t"
    "                                            \n\t"
    "vbroadcastsd       6 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       7 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd       8 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd       9 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      10 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      11 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "prefetcht0        80 * 8(%%rax)             \n\t"
    "                                            \n\t"
    "vmovapd            4 * 32(%%rbx), %%ymm0    \n\t" // iteration 2
    "vmovapd            5 * 32(%%rbx), %%ymm1    \n\t"
    "                                            \n\t"
    "vbroadcastsd      12 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      13 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd      14 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      15 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      16 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      17 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "vmovapd            6 * 32(%%rbx), %%ymm0    \n\t" // iteration 3
    "vmovapd            7 * 32(%%rbx), %%ymm1    \n\t"
    "                                            \n\t"
    "vbroadcastsd      18 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      19 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm4    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm5    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm6    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm7    \n\t"
    "                                            \n\t"
    "vbroadcastsd      20 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      21 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm8    \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm9    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm10   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm11   \n\t"
    "                                            \n\t"
    "vbroadcastsd      22 *  8(%%rax), %%ymm2    \n\t"
    "vbroadcastsd      23 *  8(%%rax), %%ymm3    \n\t"
    "vfmadd231pd       %%ymm0, %%ymm2, %%ymm12   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm2, %%ymm13   \n\t"
    "vfmadd231pd       %%ymm0, %%ymm3, %%ymm14   \n\t"
    "vfmadd231pd       %%ymm1, %%ymm3, %%ymm15   \n\t"
    "                                            \n\t"
    "addq           $4 * 6 * 8, %%rax            \n\t" // a += M_UNROLL*K_UNROLL
    "addq           $4 * 8 * 8, %%rbx            \n\t" // b += N_UNROLL*K_UNROLL
    "                                            \n\t"
    "decq   %%rsi                                \n\t" // i -= 1;
    "jnz    .DLOOPKITER                          \n\t" // iterate again if i != 0.
    "                                            \n\t"
    "vaddpd          0(%%rcx), %%ymm4, %%ymm4    \n\t"
    "vmovupd        %%ymm4,  0(%%rcx)            \n\t"
    "vaddpd         32(%%rcx), %%ymm5, %%ymm5    \n\t"
    "vmovupd        %%ymm5, 32(%%rcx)            \n\t"
    "                                            \n\t"
    "addq           %%rdi, %%rcx                 \n\t"
    "                                            \n\t"
    "vaddpd          0(%%rcx), %%ymm6, %%ymm6    \n\t"
    "vmovupd        %%ymm6,  0(%%rcx)            \n\t"
    "vaddpd         32(%%rcx), %%ymm7, %%ymm7    \n\t"
    "vmovupd        %%ymm7, 32(%%rcx)            \n\t"
    "                                            \n\t"
    "addq           %%rdi, %%rcx                 \n\t"
    "                                            \n\t"
    "vaddpd          0(%%rcx), %%ymm8, %%ymm8    \n\t"
    "vmovupd        %%ymm8,  0(%%rcx)            \n\t"
    "vaddpd         32(%%rcx), %%ymm9, %%ymm9    \n\t"
    "vmovupd        %%ymm9, 32(%%rcx)            \n\t"
    "                                            \n\t"
    "addq           %%rdi, %%rcx                 \n\t"
    "                                            \n\t"
    "vaddpd          0(%%rcx), %%ymm10, %%ymm10  \n\t"
    "vmovupd        %%ymm10,  0(%%rcx)           \n\t"
    "vaddpd         32(%%rcx), %%ymm11, %%ymm11  \n\t"
    "vmovupd        %%ymm11, 32(%%rcx)           \n\t"
    "                                            \n\t"
    "addq           %%rdi, %%rcx                 \n\t"
    "                                            \n\t"
    "vaddpd          0(%%rcx), %%ymm12, %%ymm12  \n\t"
    "vmovupd        %%ymm12,  0(%%rcx)           \n\t"
    "vaddpd         32(%%rcx), %%ymm13, %%ymm13  \n\t"
    "vmovupd        %%ymm13, 32(%%rcx)           \n\t"
    "                                            \n\t"
    "addq           %%rdi, %%rcx                 \n\t"
    "                                            \n\t"
    "vaddpd         0(%%rcx), %%ymm14, %%ymm14   \n\t"
    "vmovupd        %%ymm14,  0(%%rcx)           \n\t"
    "vaddpd         32(%%rcx), %%ymm15, %%ymm15  \n\t"
    "vmovupd        %%ymm15, 32(%%rcx)           \n\t"

    : // output operands (none)
    : // input operands
      [k]   "m" (k),
      [a]   "m" (A),
      [b]   "m" (B),
      [c]   "m" (C_ptr),
      [ldc] "m" (ldc)
    : // register clobber list
      "rax", "rbx", "rcx", "rdx", "rsi", "rdi", "r13",
      "xmm0", "xmm1", "xmm2", "xmm3",
      "xmm4", "xmm5", "xmm6", "xmm7",
      "xmm8", "xmm9", "xmm10", "xmm11",
      "xmm12", "xmm13", "xmm14", "xmm15",
      "memory"
    );
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
