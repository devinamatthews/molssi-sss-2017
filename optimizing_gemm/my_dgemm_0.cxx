#include "common.hpp"

/*
 * Compute C += A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
    /*
     * Step 0:
     *
     * Simple triple-loop matrix-matrix product.
     *
     * This ordering of the loops is called the "dot-product" algorithm.
     */
    for (int i = 0;i < m;i++)
    {
        for (int j = 0;j < n;j++)
        {
            /*
             * ...because this loop is a dot product (duh).
             */
            for (int p = 0;p < k;p++)
            {
                C(i,j) += A(i,p) * B(p,j);
            }
        }
    }
}
