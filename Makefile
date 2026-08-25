TARGET      := CUDACyclone
# CUDAHash.cu is deliberately NOT listed here. CUDACyclone.cu #includes it
# directly so the hash routines inline into the fused kernel instead of going
# through device linking. Compiling it as a second translation unit as well
# defines K, IV and verifyHash160_33_from_limbs_rare twice and the device link
# fails with "nvlink error: Multiple definition of ...". CMakeLists.txt builds
# only CUDACyclone.cu for the same reason; keep the two in step.
SRC         := CUDACyclone.cu
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
empty       :=
space       := $(empty) $(empty)
comma       := ,

KNOWN_SM_ARCHS := 75 80 86 87 88 89 90 100 101 103 110 120 121
NVCC_SM_ARCHS := $(patsubst compute_%,%,$(shell $(CC) --list-gpu-arch 2>/dev/null))
SUPPORTED_SM_ARCHS := $(filter $(NVCC_SM_ARCHS),$(KNOWN_SM_ARCHS))
ifeq ($(strip $(SUPPORTED_SM_ARCHS)),)
  SUPPORTED_SM_ARCHS := 75 86 89 90
endif

ifeq ($(OS),Windows_NT)
  DETECTED_SM_ARCHS :=
else
  DETECTED_SM_ARCHS := $(filter $(SUPPORTED_SM_ARCHS),$(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | tr -d '.' | sort -u))
endif

ifeq ($(strip $(CUDA_ARCHS)),)
  ifeq ($(strip $(PORTABLE)),1)
    SM_ARCHS := $(SUPPORTED_SM_ARCHS)
    ARCH_SOURCE := portable
  else ifneq ($(strip $(DETECTED_SM_ARCHS)),)
    SM_ARCHS := $(DETECTED_SM_ARCHS)
    ARCH_SOURCE := detected GPU
  else
    SM_ARCHS := $(SUPPORTED_SM_ARCHS)
    ARCH_SOURCE := nvcc-supported fallback
  endif
else ifeq ($(strip $(CUDA_ARCHS)),all)
  SM_ARCHS := $(SUPPORTED_SM_ARCHS)
  ARCH_SOURCE := CUDA_ARCHS=all
else
  REQUESTED_SM_ARCHS := $(subst $(comma),$(space),$(CUDA_ARCHS))
  SM_ARCHS := $(filter $(SUPPORTED_SM_ARCHS),$(REQUESTED_SM_ARCHS))
  ARCH_SOURCE := CUDA_ARCHS
endif

ifeq ($(strip $(SM_ARCHS)),)
  SM_ARCHS := $(SUPPORTED_SM_ARCHS)
  ARCH_SOURCE := fallback
endif
PTX_ARCH    := $(lastword $(SM_ARCHS))
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch)) -gencode arch=compute_$(PTX_ARCH),code=compute_$(PTX_ARCH)

$(info CUDA SM architectures ($(ARCH_SOURCE)): $(SM_ARCHS))

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE) -DCUDACYCLONE_EMBEDDED_CPU_WORKER
CXXFLAGS   := -std=c++17

ifeq ($(OS),Windows_NT)
  RM := cmd /C del /Q
  EXE := .exe
  # cpu_avx2/ gates its MSVC vs GCC intrinsic paths on WIN64, not the _WIN64
  # that MSVC predefines. Without this, Int.h(197) falls into the GCC branch
  # and redefines _umul128/_mul128/_udiv128 plus __builtin_ia32_addcarryx_u64,
  # which MSVC rejects. CMakeLists.txt defines WIN64 for the same reason; keep
  # the two in step.
  NVCC_FLAGS += -DWIN64
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
