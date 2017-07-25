# Optimizing BLAS

### (on x86 architectures at least)

This tutorial illustrates some basic and some not-so-basic optimization techniques in the context of optimizing the GEMM (GEneral Matrix-Matrix multiply) routine from the Basic Linear Algebra Subprograms (BLAS). C++11 is used for the implementation.

The particular form of GEMM that will be implemented is the operation:

`C += A*B`

for general matrices `A`, `B`, and `C`. It is assumed (and enforced in the example program) that the matrices are row-major. The Eigen library is used to simplify some of the high-level handling of matrices, and to provide a check on the correctness and performance of the implementation (the Eigen library uses an external BLAS library, assumed to be OpenBLAS in the default Makefile).

## References

- Anatomy of high-performance matrix multiplication. Kazushige Goto, Robert A. van de Geijn. ACM Transactions on Mathematical Software (TOMS), 2008

- BLIS: A Framework for Rapidly Instantiating BLAS Functionality. Field G. Van Zee, Robert A. van de Geijn. ACM Transactions on Mathematical Software (TOMS), 2015.

  (Both available without charge at: http://www.cs.utexas.edu/users/flame/FLAMEPublications.html)

- BLISlab: A Sandbox for Optimizing GEMM (https://github.com/flame/blislab)

This tutorial is largely based on:

- "How to Optimize GEMM", J. Huang and R.A. van de Geijn (https://github.com/flame/how-to-optimize-gemm/wiki)

## Step 0: Set-up and the triple loop

Prerequisites:
 - Unix or Linux OS (including OS X/macOS and Cygwin)
 - g++ (version 4.7 or later---another C++11 compiler may be used if set up in the Makefile)
 - OpenMP support (default in g++)
 - OpenBLAS (may also be changed in the Makefile)
 - Eigen v3
 - gnuplot
 - ghostscript
 - Intel or AMD x86-64 processor with AVX (for the last example only, FMA support is required)

In an anaconda environment, the additional requirements can be installed with:

```bash
conda install -c menpo -c bioconda eigen=3.2.7 gnuplot=4.6 openblas
conda install -c conda-forge ghostscript
```

The various example implementations of GEMM are contained in the files `my_dgemm_<n>.cxx` where `n` is from 0 to 8. To compile and automatically run and plot the first "triple-loop" example, run the command:

```
make STEP=0
```

Assuming everything went correctly, you should see a plot like this:

![Step 0](figures/step0.png?raw=true)

In each of the following steps, the example file will be built and run with `make STEP=<n>`. The default plot will show the current performance, the performance of the previous step, and BLAS. To plot other results, use `./plot.sh <n1> <n2> <n3> ...`.

The example files also contain comments describing the optimizations performed at that step in more detail, as well as some brain-teaser exercises. So far, performance is pretty poor... so let's get optimizing!

## Step 1: Investigating different loop orders

In this step, we see what happens if we perform the three loops in a different order. After running the example for this step, you should see something like:

![Step 1](figures/step1.png?raw=true)

This new implementation is perhaps slightly faster for very small matrices, but is generally slower. (Why? Check out the comments!)

## Step 2: Unrolling

In this step, two of the loops are unrolled. This requires splitting the loop into two loops, where the inner loop has a fixed number of iterations. Then, the inner iterations are explicitly copied out. Now we start to see some improvement:

![Step 2](figures/step2.png?raw=true)

Of course, we went a bit backwards in that last step, so lets see everything so far:

![Steps 0, 1, and 2](figures/step012.png?raw=true)

Not bad, but lots of room for improvement.

## Step 3: Helping the compiler out

Compilers are great, but sometimes they need help. In this step, we implement our unrolled GEMM kernel using pointer arithmetic and temporary storage space for `C`. This helps a bit:

![Step 3](figures/step3.png?raw=true)

## Step 4: Manual vectorization with intrinsics

Ideally the compiler would really run with the code we gave it in the previous step. In fact, some really fancy compilers like Intel's can likely vectorize Step 3 all by themselves. Unfortunately, GCC isn't quite that good (yet). We can tell that GCC hasn't done any vectorization by looking at the assembly it generates:

![Disassembly of Step 3](figures/disassembly-novec.png?raw=true)

We've zoomed in on part of the loop body, where we can see that it is doing `vfmadd231sd` instructions. the `vfmadd` part indicates that it is doing a multiplication (`A*B`) and an add (`C +=`) at the same time. But, the `sd` part shows that it is only computing this for a **s**ingle **d**ouble-precision element, and not a whole vector.

Instead, let's use vector intrinsics to tell it exactly what we want to do. Check out the code for more details on this step. Now that we've hand-vectorized the code, is the compiler doing what we want? Let's look at the new assembly:

![Disassembly of Step 4](figures/disassembly-vec.png?raw=true)

Now we see a much tighter disassembly (in fact, this is the entire loop body), including `vfmadd231pd` (**p**acked **d**ouble) instructions as well as `vbroadcastsd` (which despite having an "s" is a good vector instruction). The performance is *much* better too:

![Step 4](figures/step4.png?raw=true)

## Step 5: Blocking for cache

Although the manually vectorized code is very fast for small matrices (even faster than BLAS sometimes!), it falls behind for larger matrices. This is because the kernel has to wait while data is read in. If the data is not in the cache (preferably the lowest-level cache L1), then it may have to wait a long time, hurting performance. To keep data in cache better, the loops can be blocked. In this example, multiple loops are blocked such that various pieces of the matrices are kept in the L1, L2, and L3 caches:

![Step 5](figures/step5.png?raw=true)

We can see that performance starts to improve as we get to large matrices (`N ~= 1000`). The amount of improvement increases for even larger problem sizes. However, enabling the full benefit of blocking requires further optimizations.

## Step 6: Packing data for optimal reuse

Even though the *amount* of data we are reusing should fit in the caches thanks to blocking, it may be prematurely evicted by something else, or require too much bandwidth to the processor (when the data is not nicely packed into cache lines). To fix this we pack blocks of `A` and `B` into contiguous, stream-lined storage.

Sure, this is extra work, but the cost is *amortized* over the number of times we reuse the data. It also just happens to help performance quite a bit:

![Step 6](figures/step6.png?raw=true)

## Step 7: Prefetching and more unrolling

Now we are getting fairly close to BLAS-level performance. The optimizations remaining to do are getting progressively more complex and offer less of a return on performance. But, a little more optimization never hurt...

In this step, we apply prefetching for elements of `A` and `C` in the micro-kernel. We also unroll the loop in the micro-kernel. Does this help performance?

![Step 7](figures/step7.png?raw=true)

Whoa! What happened? Sometimes optimizations backfire, but luckily this time it is the compiler's fault and not the programmer. What happened is that GCC has reordered all of our intrinsic operations in a way that severely limits performance.

## Step 8: Side-stepping the compiler: Inline assembly

Sometimes it is necessary to forego the compiler altogether and just tell the processor exactly what we want it to do. Ideally, intrinsics would get us all the way but every once in a while they are just not enough. In this step, we translate our intrinsics code to inline assembly, but don't otherwise apply any new optimizations. (Note: this step requires a processor with FMA instructions)

![Step 8](figures/step8.png?raw=true)

Wow, we definitely got back a lot of performance from our previous pitfall, but lets look a little further back too, to make sure the prefetching and unrolling optimizations actually paid off:

![Steps 6, 7, and 8](figures/step678.png?raw=true)

Indeed they did. The last little bit of performance difference between this code and a highly-tuned BLAS is all in the details. Be prepared to spend several years optimizing if you really want that last little bit.

Here is how our fully-optimized implementation compares to the simple triple-loop:

![Steps 0 and 8](figures/step08.png?raw=true)

Night and day!
