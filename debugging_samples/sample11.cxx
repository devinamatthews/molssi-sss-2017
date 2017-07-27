#include <vector>
#include <cassert>

double get_matrix_element(const double* array, int m, int n, int i, int j, int lda)
{
    assert(i >= 0 && i < m);
    assert(j >= 0 && j < n);
    return array[i+j*lda];
}

int main()
{
    int m = 10;
    int n = 100;
    std::vector<double> matrix(m*n);
    get_matrix_element(matrix.data(), m, n, 11, 99, m);
}
