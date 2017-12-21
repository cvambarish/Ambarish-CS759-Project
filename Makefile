# Warnings
WFLAGS := -Wall -Wextra -Wsign-conversion -Wsign-compare

# Optimization and architecture
OPT  := -O3 -mavx -fopenmp
OPT2		:= -O3 
# Language standard
CCSTD	:= -std=c99
CXXSTD	:= -std=c++14

# Linker options
LDFLAGS := -fopenmp

# Names of executables to create
EXEC := PoissonEquationJacobi PoissonEquationJacobiOMP

.DEFAULT_GOAL := all

BIN = "/usr/local/gcc/6.4.0/bin/gcc"

.PHONY: debug
debug : OPT  := -O0 -g -fno-omit-frame-pointer -fsanitize=address
debug : LDFLAGS := -fsanitize=address
debug : $(EXEC)

all : Makefile $(EXEC) jacobiCuda

%.o : %.c Makefile
	@ echo Compiling $<...
	@ $(CC) $(CCSTD) $(WFLAGS) -Wno-unused-result $(OPT) $(CFLAGS) -c $< -o $@

%.o : %.cpp Makefile
	@ echo Compiling $<...
	@ $(CXX) $(CXXSTD) $(WFLAGS) $(OPT) $(CXXFLAGS) -c $< -o $@

$(EXEC) : % : %.o
	@ echo Building $@...
	@ $(CXX) -o $@ $< $(LDFLAGS)

jacobiCuda: PoissonEquationJacobiCuda.cu
	module load cuda;nvcc -o PoissonEquationJacobiCuda $(OPT2) PoissonEquationJacobiCuda.cu -ccbin $(BIN)
	
	
.PHONY: clean
clean:
	@ rm -f *.o $(EXEC) *.in
