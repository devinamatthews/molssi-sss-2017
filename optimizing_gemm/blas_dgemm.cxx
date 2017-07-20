#include "common.hpp"

#include "lawrap/blas.h"
using namespace LAWrap;

/*
 * Compute C = A*B
 */
void my_dgemm(int m, int n, int k, const matrix& A, const matrix& B, matrix& C)
{
    gemm('N', 'N', n, m, k,
         1.0, B.data(), n,
              A.data(), k,
         1.0, C.data(), n);
}
