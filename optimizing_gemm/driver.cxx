#include "common.hpp"

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
    for (int n = 24;n <= 1008;n += 24)
    {
        matrix A(n,n), B(n,n), C(n,n);

        // time DGEMM on matrices of zeros (doesn't affect speed)

        double elapsed = benchmark([&] { my_dgemm(n, n, n, A, B, C); }, 5);
        double flops = 2.0*n*n*n;
        double gflops = flops/elapsed/1e9;

        printf("%d %f\n", n, gflops);
        fflush(stdout);

        // check accuracy for random matrices

        A.setRandom();
        B.setRandom();
        C.setRandom();

        matrix ABplusC = C;
        my_dgemm(n, n, n, A, B, ABplusC);

        ABplusC -= (A*B+C);
        double error = ABplusC.norm();
        double error_limit = 2*std::numeric_limits<double>::epsilon()*n*n;

        if (error > error_limit)
        {
            fprintf(stderr, "error limit exceeded for n=%d: %g > %g\n",
                    n, error, error_limit);
        }
    }

    return 0;
}
