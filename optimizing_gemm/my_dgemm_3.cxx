#include "common.hpp"

#define M_UNROLL 6
#define N_UNROLL 8

/*
 * Compute C += A*B for some subblocks of A, B, and C
 */
template <typename MatrixA, typename MatrixB, typename MatrixC>
void my_dgemm_kernel(int k, const MatrixA& A, const MatrixB& B, MatrixC& C)
{
    /*
     * Step 3:
     *
     * Now that we've unrolled the i and j loops, we need to help the compiler
     * optimize the loop body. Some problems that prohibit optimization:
     *
     * 1) Aliasing: when we write to C(i,j), the compiler has no way of
     *    knowing that this isn't the same location as one of the A(i,p) or
     *    B(p,j) elements, and so it has to always re-load them from memory
     *    instead of making a copy in registers.
     *
     *    In this step, we make a temporary area to write to (C_tmp) instead,
     *    that the compiler knows is safe.
     *
     * 2) Function inlining: the expression A(i,p) is actually a function call!
     *    It may actually be a whole sequence of function calls. Often, the
     *    compiler can inline these small function calls and optimize the
     *    result, but some optimizations are still prevented.
     *
     *    In this step, we use explicit pointer arithmetic to access the
     *    elements of A and B.
     *
     * 3) Indexing arithmetic: even after turning accesses like A(i,p) into
     *    pointer arithmetic, we still have to do a lot of index math to figure
     *    out where each index is.
     *
     *    In this example, we apply strengh reduction to simplify the pointer
     *    arithmetic (especially for B, hint hint).
     *
     * Exercise: the possibility of aliasing can still prevent compiler
     * optimizations in some cases. What is another way to tell the compiler
     * that pointers do not alias each other?
     */
    const double* A_ptr = A.data();
    const double* B_ptr = B.data();

    /*
     * NB: &A(i,p) = A.data() + i*lda + p and similarly for B
     */
    int lda = A.outerStride();
    int ldb = B.outerStride();

    double C_tmp[M_UNROLL][N_UNROLL] = {};

    for (int p = 0;p < k;p++)
    {
        double A_value = A_ptr[0*lda];
        C_tmp[0][0] += A_value * B_ptr[0];
        C_tmp[0][1] += A_value * B_ptr[1];
        C_tmp[0][2] += A_value * B_ptr[2];
        C_tmp[0][3] += A_value * B_ptr[3];
        C_tmp[0][4] += A_value * B_ptr[4];
        C_tmp[0][5] += A_value * B_ptr[5];
        C_tmp[0][6] += A_value * B_ptr[6];
        C_tmp[0][7] += A_value * B_ptr[7];

        A_value = A_ptr[1*lda];
        C_tmp[1][0] += A_value * B_ptr[0];
        C_tmp[1][1] += A_value * B_ptr[1];
        C_tmp[1][2] += A_value * B_ptr[2];
        C_tmp[1][3] += A_value * B_ptr[3];
        C_tmp[1][4] += A_value * B_ptr[4];
        C_tmp[1][5] += A_value * B_ptr[5];
        C_tmp[1][6] += A_value * B_ptr[6];
        C_tmp[1][7] += A_value * B_ptr[7];

        A_value = A_ptr[2*lda];
        C_tmp[2][0] += A_value * B_ptr[0];
        C_tmp[2][1] += A_value * B_ptr[1];
        C_tmp[2][2] += A_value * B_ptr[2];
        C_tmp[2][3] += A_value * B_ptr[3];
        C_tmp[2][4] += A_value * B_ptr[4];
        C_tmp[2][5] += A_value * B_ptr[5];
        C_tmp[2][6] += A_value * B_ptr[6];
        C_tmp[2][7] += A_value * B_ptr[7];

        A_value = A_ptr[3*lda];
        C_tmp[3][0] += A_value * B_ptr[0];
        C_tmp[3][1] += A_value * B_ptr[1];
        C_tmp[3][2] += A_value * B_ptr[2];
        C_tmp[3][3] += A_value * B_ptr[3];
        C_tmp[3][4] += A_value * B_ptr[4];
        C_tmp[3][5] += A_value * B_ptr[5];
        C_tmp[3][6] += A_value * B_ptr[6];
        C_tmp[3][7] += A_value * B_ptr[7];

        A_value = A_ptr[4*lda];
        C_tmp[4][0] += A_value * B_ptr[0];
        C_tmp[4][1] += A_value * B_ptr[1];
        C_tmp[4][2] += A_value * B_ptr[2];
        C_tmp[4][3] += A_value * B_ptr[3];
        C_tmp[4][4] += A_value * B_ptr[4];
        C_tmp[4][5] += A_value * B_ptr[5];
        C_tmp[4][6] += A_value * B_ptr[6];
        C_tmp[4][7] += A_value * B_ptr[7];

        A_value = A_ptr[5*lda];
        C_tmp[5][0] += A_value * B_ptr[0];
        C_tmp[5][1] += A_value * B_ptr[1];
        C_tmp[5][2] += A_value * B_ptr[2];
        C_tmp[5][3] += A_value * B_ptr[3];
        C_tmp[5][4] += A_value * B_ptr[4];
        C_tmp[5][5] += A_value * B_ptr[5];
        C_tmp[5][6] += A_value * B_ptr[6];
        C_tmp[5][7] += A_value * B_ptr[7];

        A_ptr++;
        B_ptr += ldb;
    }

    for (int i = 0;i < M_UNROLL;i++)
    {
        for (int j = 0;j < N_UNROLL;j++)
        {
            C(i,j) += C_tmp[i][j];
        }
    }
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
