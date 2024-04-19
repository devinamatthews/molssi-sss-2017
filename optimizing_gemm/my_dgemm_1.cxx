#include "common.hpp"

/*
 * Compute C += A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
    /*
     * Step 1:
     *
     * Variant triple-loop matrix-matrix product.
     *
     * This ordering of is the "outer-product" algorithm.
     */
    for (int p = 0;p < k;p++)
    {
        /*
         * ...because these two loops perform an outer product (rank-1 update).
         */
        for (int i = 0;i < m;i++)
        {
            /*
             * Here, we put the "j" loop innermost. This is because elements of B
             * and C in the n dimension are contiguous, and so we will benefit from
             * Re-using elements in the same cache line which have already been
             * loaded into the cache.
             *
             * Note that different loop orderings can have a large effect on
             * performance, and may hurt or help depending on the problem size.
             * Generally, you should prefer loop orderings that have contiguous
             * access in the innermost loop.
             *
             * Exercise: what is the maximum reduction in bandwidth when we
             * reorder the loops to ensure contiguous access?
             */
            for (int j = 0;j < n;j++)
            {
                C(i,j) += A(i,p) * B(p,j);
            }
        }
    }
}
