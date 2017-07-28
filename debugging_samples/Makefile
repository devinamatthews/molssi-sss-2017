CXX=g++
CXXFLAGS=-std=c++11 -g -O0
LDFLAGS=-Wl,-rpath,$(CONDA_PREFIX)/lib

all:

%.o: %.cxx
	$(CXX) -c -o $@ $^ $(CXXFLAGS)

%.x: %.o
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm *.x *.o

.SECONDARY:

