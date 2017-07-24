#include "common.hpp"

#include <lawrap/blas.h>

/*
 * Compute C += A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
    // BLAS assumes that the matrices are column-major, but
    // out matrices are row-major. Since A (row-major) =
    // A^T (column-major), we can use the C^T += B^T A^T instead
    // of C += A*B. Note that this just means we switch "A"
    // with "B" and "m" with "n".
    LAWrap::gemm('N', 'N', n, m, k,
                 1.0, B.data(), n,
                      A.data(), k,
                 1.0, C.data(), n);

}

