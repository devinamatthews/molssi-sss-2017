SHELL=/bin/bash

CXX=g++
CXXFLAGS=-O3 -march=native -fopenmp
LDFLAGS=-fopenmp -lopenblas

all: run

%.o: %.cxx
	$(CXX) -c -o $@ $^ $(CXXFLAGS)
	
blas.x: driver.o blas_dgemm.o
	$(CXX) -o $@ $^ $(LDFLAGS)
	
driver.x: driver.o my_dgemm_$(STEP).o
	$(CXX) -o $@ $^ $(LDFLAGS)
	
run: blas.x driver.x
	@echo "Running BLAS DGEMM"
	@./blas.x | tee out_blas
	@echo "Running DGEMM from Step $(STEP)"
	@./driver.x | tee out_$(STEP)
	@if [ $(STEP) = 0 ]; then \
		./plot.sh $(STEP); \
	else \
		./plot.sh $$((STEP-1)) $(STEP); \
	fi
	
