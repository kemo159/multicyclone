TARGET      := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
CPU_SRC     := \
	cpu_avx2/Cyclone.cpp \
	cpu_avx2/SECP256K1.cpp \
	cpu_avx2/Int.cpp \
	cpu_avx2/IntGroup.cpp \
	cpu_avx2/IntMod.cpp \
	cpu_avx2/Point.cpp \
	cpu_avx2/ripemd160_avx2.cpp \
	cpu_avx2/p2pkh_decoder.cpp \
	cpu_avx2/sha256_avx2.cpp \
	cpu_avx2/Random.cpp \
	cpu_avx2/Timer.cpp
OBJ         := $(SRC:.cu=.o) $(CPU_SRC:.cpp=.o)
CC          := nvcc

KNOWN_SM_ARCHS := 75 80 86 87 88 89 90 100 101 103 110 120 121
NVCC_SM_ARCHS := $(patsubst compute_%,%,$(shell $(CC) --list-gpu-arch 2>/dev/null))
SM_ARCHS   := $(filter $(NVCC_SM_ARCHS),$(KNOWN_SM_ARCHS))
ifeq ($(strip $(SM_ARCHS)),)
  SM_ARCHS := 75 86 89 90
endif
PTX_ARCH    := $(lastword $(SM_ARCHS))
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch)) -gencode arch=compute_$(PTX_ARCH),code=compute_$(PTX_ARCH)

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE) -DCUDACYCLONE_EMBEDDED_CPU_WORKER
CXXFLAGS   := -std=c++17

ifeq ($(OS),Windows_NT)
  RM := cmd /C del /Q
  EXE := .exe
  HOST_FLAGS := -Xcompiler "/EHsc /O2 /arch:AVX2 /openmp"
  LDFLAGS := Advapi32.lib -lcudadevrt -cudart=static
else
  RM := rm -f
  EXE := 
  HOST_FLAGS := -Xcompiler "-O3 -mavx2 -mbmi2 -madx -funroll-loops -fopenmp"
  LDFLAGS := -lcudadevrt -cudart=static -lgomp -lpthread
endif

all: $(TARGET)$(EXE)

$(TARGET)$(EXE): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(HOST_FLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(HOST_FLAGS) -c $< -o $@

%.o: %.cpp
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(HOST_FLAGS) -c $< -o $@

clean:
ifeq ($(OS),Windows_NT)
	-$(RM) $(TARGET)$(EXE) $(OBJ) 2>nul
else
	-$(RM) $(TARGET)$(EXE) $(OBJ) 2>/dev/null
endif
