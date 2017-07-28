################################################################################
# Fill these out for your own project
################################################################################

# The name of your executable
PROGRAM:=

# A good default
CXX:=g++

# Add whatever flags you need
CXXFLAGS:=-std=c++11

# Add whatever flags or external libraries you need
LDFLAGS:=-Wl,-rpath,$(CONDA_PREFIX)/lib

################################################################################
# From here on is "boilerplate" alter at your own risk
################################################################################

SOURCES:=$(shell find . -iname "*.cxx" -o -iname "*.cpp" -o -iname "*.cc")
OBJECTS:=$(addsuffix .o, $(basename $(SOURCES))))

all: $(PROGRAM)

%.cxx: %.o
	$(CXX) -c -o $@ $^ $(CXXFLAGS)

%.cpp: %.o
	$(CXX) -c -o $@ $^ $(CXXFLAGS)

%.cc: %.o
	$(CXX) -c -o $@ $^ $(CXXFLAGS)
	
$(PROGRAM): $(OBJECTS)
	$(CXX) -o $@ $(OBJECTS) $(LDFLAGS)
	
clean:
	find . -name '*.o' -delete
	rm -f $(PROGRAM)
	
