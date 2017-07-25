#include <cstdio>
#include <limits>
#include <algorithm>
#include <vector>

#include <omp.h>

void transpose(int m, int n,
               const double* A, int lda,
                     double* B, int ldb);

template <typename Experiment>
double benchmark(Experiment&& exp, int num_repeat=1)
{
    double min_time = std::numeric_limits<double>::max();

    for (int i = 0;i < num_repeat;i++)
    {
        double t0 = omp_get_wtime();
        exp();
        double t1 = omp_get_wtime();
        min_time = std::min(min_time, t1-t0);
    }

    return min_time;
}

int main(int argc, char** argv)
{
    int nmax = 1000;

    std::vector<double> A(nmax*nmax), B(nmax*nmax);

    for (int n = 4;n <= nmax;n += 4)
    {
        double elapsed = benchmark([&] { transpose(n, n, A.data(), n, B.data(), n); }, 100);
        double bytes = 2*sizeof(double)*n*n;
        double gbps = bytes/elapsed/1024/1024/1024;

        printf("%d %g\n", n, gbps);
        fflush(stdout);
    }

    return 0;
}
