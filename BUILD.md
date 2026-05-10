# Build Instructions

This project builds on native Windows and Linux with NVIDIA CUDA. CUDA Toolkit 13.x is recommended for current RTX 50-series and newer CUDA 13 architecture support. The main executable also embeds the AVX2 CPU worker used by `--cpu-auto`.

## Windows

### Required Software

Install:

- NVIDIA driver with CUDA support
- CUDA Toolkit 13.x
- Visual Studio 2022 Build Tools
- CMake 3.24 or newer
- AVX2-capable CPU

When installing Visual Studio Build Tools, select:

- Desktop development with C++
- MSVC v143 C++ build tools
- Windows 10 or Windows 11 SDK
- C++ CMake tools for Windows

Verify the tools from PowerShell:

```powershell
nvcc --version
cmake --version
cl
```

The `cl` command should print Microsoft compiler information. If PowerShell cannot find `cl`, open "x64 Native Tools Command Prompt for VS 2022" and run the build commands there.

### Build

From the project directory:

```powershell
cmake -S . -B build-msvc -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-msvc --config Release --parallel
```

The executable will be created at:

```text
build-msvc\Release\CUDACyclone.exe
```

Run a smoke test:

```powershell
.\build-msvc\Release\CUDACyclone.exe --help
```

Hybrid CPU/GPU smoke test:

```powershell
.\build-msvc\Release\CUDACyclone.exe --range 200000000:2000003ff --address 17Dw58gss9JDWWjjsPtT8Tts6T3aeDTMhx --grid 128,8 --slices 1 --cpu-threads 1 --cpu-percent 25
```

### Choosing CUDA Architectures

Use the architecture that matches your GPU:

```text
RTX 30xx: 86
RTX 40xx: 89
Hopper/H100: 90
Blackwell/RTX 50xx: 120 or 121
```

Examples:

```powershell
cmake -S . -B build-msvc -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake -S . -B build-msvc -A x64 -DCMAKE_CUDA_ARCHITECTURES="89;120"
```

## Linux

### Required Packages

Install:

- NVIDIA driver with CUDA support
- CUDA Toolkit 13.x
- GCC/G++
- Make
- CMake, optional but recommended
- AVX2/BMI2/ADX-capable CPU

Ubuntu/Debian package basics:

```bash
sudo apt update
sudo apt install -y build-essential make cmake
```

Install the CUDA Toolkit from NVIDIA's official repository for your distribution. After installation, verify:

```bash
nvcc --version
nvidia-smi
make --version
cmake --version
```

If `nvcc` is not found, add CUDA to your shell path. Adjust the CUDA version directory if needed:

```bash
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

You can add those two lines to `~/.bashrc` if they are needed every time.

### Build With Make

From the project directory:

```bash
make clean
make -j"$(nproc)"
```

The executable will be created at:

```text
./CUDACyclone
```

Run a smoke test:

```bash
./CUDACyclone --help
```

The Makefile builds the CUDA pipeline and the embedded AVX2 CPU worker into one binary.

### Build With CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel
```

The executable will be created at:

```text
build/CUDACyclone
```

Run a hybrid CPU/GPU smoke test:

```bash
./build/CUDACyclone --range 200000000:2000003ff --address 17Dw58gss9JDWWjjsPtT8Tts6T3aeDTMhx --grid 128,8 --slices 1 --cpu-threads 1 --cpu-percent 25
```

## WSL2 Notes

For WSL2 on Windows:

- Install the NVIDIA Windows driver that supports CUDA on WSL.
- Install Ubuntu from the Microsoft Store.
- Install the Linux CUDA Toolkit inside WSL.
- Build using the Linux instructions above.
- `--cpu-auto` works in WSL as long as the binary can relaunch itself and write `cpu_worker.log`/`cpu_worker.stats` in the current directory.

Verify GPU visibility inside WSL:

```bash
nvidia-smi
nvcc --version
```

If `nvidia-smi` works but `nvcc` does not, the driver is visible but the CUDA Toolkit is not installed inside WSL.

## Cleaning

Makefile build:

```bash
make clean
```

CMake build:

```bash
rm -rf build
```

Windows CMake build:

```powershell
Remove-Item -Recurse -Force build-msvc
```

Runtime files such as `found_key.txt`, `found_keys.txt`, `cpu_worker.log`, and `cpu_worker.stats` can be deleted between benchmark or smoke-test runs.
