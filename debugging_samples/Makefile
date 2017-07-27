CXX=g++
CXXFLAGS=-std=c++11 -g -O0
LDFLAGS=-Wl,-rpath,$(HOME)/miniconda/envs/sss/lib

all:

%.o: %.cxx
	$(CXX) -c -o $@ $^ $(CXXFLAGS)

%.x: %.o
	$(CXX) -o $@ $^ $(LDFLAGS)

clean:
	rm *.x *.o

.SECONDARY:

