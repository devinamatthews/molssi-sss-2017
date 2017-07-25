SHELL=/bin/bash

#OpenBLAS
#BLAS_LIBS=-lopenblas
#MKL
BLAS_LIBS=-lmkl_intel_lp64 -lmkl_sequential -lmkl_core

CXX=g++
CXXFLAGS=-std=c++11 -O3 -march=native -fopenmp -DNDEBUG -I$(HOME)/miniconda/envs/sss/include -Wno-depracated-declarations
LDFLAGS=-fopenmp $(BLAS_LIBS) -Wl,-rpath,$(HOME)/miniconda/envs/sss/lib

all: run

%.o: %.cxx
	$(CXX) -c -o $@ $^ $(CXXFLAGS)
	
out_blas: driver.o blas_dgemm.o
	$(CXX) -o blas.x $^ $(LDFLAGS)
	@echo "Running BLAS DGEMM"
	@./blas.x | tee out_blas

out_%: driver.o my_dgemm_%.o out_blas
	$(CXX) -o driver.x $(filter-out out_blas,$^) $(LDFLAGS)
	@echo "Running DGEMM from Step $(STEP)"
	@./driver.x | tee out_$(STEP)
	
run: out_$(STEP) my_dgemm_$(STEP).o
	@if [ $(STEP) = 0 ]; then \
		./plot.sh $(STEP); \
	else \
		./plot.sh $$((STEP-1)) $(STEP); \
	fi

clean:
	rm *.o *.x out_* plot_*

