#include "common.hpp"

#define M_UNROLL 6
#define N_UNROLL 8

/*
 * Compute C += A*B for some subblocks of A, B, and C
 */
template <typename MatrixA, typename MatrixB, typename MatrixC>
void my_dgemm_kernel(int k, const MatrixA& A, const MatrixB& B, MatrixC& C)
{
    for (int p = 0;p < k;p++)
    {
        C(0,0) += A(0,p) * B(p,0); C(0,1) += A(0,p) * B(p,1);
        C(0,2) += A(0,p) * B(p,2); C(0,3) += A(0,p) * B(p,3);
        C(0,4) += A(0,p) * B(p,4); C(0,5) += A(0,p) * B(p,5);
        C(0,6) += A(0,p) * B(p,6); C(0,7) += A(0,p) * B(p,7);

        C(1,0) += A(1,p) * B(p,0); C(1,1) += A(1,p) * B(p,1);
        C(1,2) += A(1,p) * B(p,2); C(1,3) += A(1,p) * B(p,3);
        C(1,4) += A(1,p) * B(p,4); C(1,5) += A(1,p) * B(p,5);
        C(1,6) += A(1,p) * B(p,6); C(1,7) += A(1,p) * B(p,7);

        C(2,0) += A(2,p) * B(p,0); C(2,1) += A(2,p) * B(p,1);
        C(2,2) += A(2,p) * B(p,2); C(2,3) += A(2,p) * B(p,3);
        C(2,4) += A(2,p) * B(p,4); C(2,5) += A(2,p) * B(p,5);
        C(2,6) += A(2,p) * B(p,6); C(2,7) += A(2,p) * B(p,7);

        C(3,0) += A(3,p) * B(p,0); C(3,1) += A(3,p) * B(p,1);
        C(3,2) += A(3,p) * B(p,2); C(3,3) += A(3,p) * B(p,3);
        C(3,4) += A(3,p) * B(p,4); C(3,5) += A(3,p) * B(p,5);
        C(3,6) += A(3,p) * B(p,6); C(3,7) += A(3,p) * B(p,7);

        C(4,0) += A(4,p) * B(p,0); C(4,1) += A(4,p) * B(p,1);
        C(4,2) += A(4,p) * B(p,2); C(4,3) += A(4,p) * B(p,3);
        C(4,4) += A(4,p) * B(p,4); C(4,5) += A(4,p) * B(p,5);
        C(4,6) += A(4,p) * B(p,6); C(4,7) += A(4,p) * B(p,7);

        C(5,0) += A(5,p) * B(p,0); C(5,1) += A(5,p) * B(p,1);
        C(5,2) += A(5,p) * B(p,2); C(5,3) += A(5,p) * B(p,3);
        C(5,4) += A(5,p) * B(p,4); C(5,5) += A(5,p) * B(p,5);
        C(5,6) += A(5,p) * B(p,6); C(5,7) += A(5,p) * B(p,7);
    }
}

/*
 * Compute C += A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
    /*
     * Step 2:
     *
     * The first real optimization is loop unrolling.
     *
     * We apply this optimization in two steps:
     *
     * First: make a copy of each loop to unroll, where the inner loop has
     *        a fixed count N, and the outer loop increments in steps of N.

    for (int p = 0;p < k;p++)
    {
        for (int j = 0;j < n;j++)
        {
            for (int i = 0;i < m;i++)
            {
                //loop body
            }
        }
    }

     * becomes

    for (int p = 0;p < k;p++)
    {
        for (int j = 0;j < n;j += N_UNROLL)
        {
            for (int i = 0;i < m;i += M_UNROLL)
            {
                for (int j = 0;j < N_UNROLL;j++)
                {
                    for (int i = 0;i < M_UNROLL;i++)
                    {
                        //loop body
                    }
                }
            }
        }
    }

     * Second: the inner loops with a fixed count are fully expanded. In this
     *         example, we've also changed the order of the loops slight and
     *         moved the inner remaining loop (over p) into a kernel.

    for (int p = 0;p < k;p++)
    {
        for (int j = 0;j < n;j += N_UNROLL)
        {
            for (int i = 0;i < m;i += M_UNROLL)
            {
                //loop body M_UNROLL*N_UNROLL times
            }
        }
    }

     * Some times the compiler can do this optimization automatically, or
     * with the help of hints such as #pragma unroll (icc/icpc) or
     * -funroll-loops (gcc/g++).
     *
     * Exercise: what happens when e.g. m%M_UNROLL != 0? How should the code
     * be amended to address this case?
     */
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
