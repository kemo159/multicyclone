TARGET      := CUDACyclone
SRC         := CUDACyclone.cu CUDAHash.cu
OBJ         := $(SRC:.cu=.o)
CC          := nvcc

KNOWN_SM_ARCHS := 75 80 86 87 88 89 90 100 101 103 110 120 121
NVCC_SM_ARCHS := $(patsubst compute_%,%,$(shell $(CC) --list-gpu-arch 2>/dev/null))
SM_ARCHS   := $(filter $(NVCC_SM_ARCHS),$(KNOWN_SM_ARCHS))
ifeq ($(strip $(SM_ARCHS)),)
  SM_ARCHS := 75 86 89 90
endif
PTX_ARCH    := $(lastword $(SM_ARCHS))
GENCODE    := $(foreach arch,$(SM_ARCHS),-gencode arch=compute_$(arch),code=sm_$(arch)) -gencode arch=compute_$(PTX_ARCH),code=compute_$(PTX_ARCH)

NVCC_FLAGS := -O3 -rdc=true -use_fast_math --ptxas-options=-O3 $(GENCODE)
CXXFLAGS   := -std=c++17

ifeq ($(OS),Windows_NT)
  RM := cmd /C del /Q
  EXE := .exe
  LDFLAGS := -lcudadevrt -cudart=static
else
  RM := rm -f
  EXE := 
  LDFLAGS := -lcudadevrt -cudart=static -lpthread
endif

all: $(TARGET)$(EXE)

$(TARGET)$(EXE): $(OBJ)
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) $(OBJ) -o $@ $(LDFLAGS)

%.o: %.cu
	$(CC) $(NVCC_FLAGS) $(CXXFLAGS) -c $< -o $@

clean:
ifeq ($(OS),Windows_NT)
	-$(RM) $(TARGET)$(EXE) $(OBJ) 2>nul
else
	-$(RM) $(TARGET)$(EXE) $(OBJ) 2>/dev/null
endif
