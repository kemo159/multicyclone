# MultiCyclone

MultiCyclone is a fork of CUDACyclone, a CUDA-based GPU Satoshi puzzle solver. This fork keeps the original high-throughput secp256k1 and Hash160 search pipeline, and adds native Windows build support, Linux/WSL build verification, CUDA 13.x architecture updates, and cleaner packaging/build documentation.

This software is intended for legitimate puzzle, research, and recovery workflows where you are authorized to search the key range.

## Credits

This project is forked from Dookoo2's CUDACyclone work.

Secp256k1 math is based on the excellent work from:

- [JeanLucPons/VanitySearch](https://github.com/JeanLucPons/VanitySearch)
- [FixedPaul/VanitySearch-Bitcrack](https://github.com/FixedPaul)

Special thanks to Jean-Luc Pons and the original CUDACyclone author for the foundational CUDA and cryptographic work.

## What This Fork Adds

- Native Windows build support with Visual Studio 2022, CMake, and CUDA.
- Linux build support through the existing Makefile and new CMake configuration.
- WSL2 build/runtime smoke testing.
- CUDA 13.x-oriented architecture support, including Blackwell-era compute 12.x targets.
- Portable host/device arithmetic fixes for MSVC compatibility.
- Dedicated [BUILD.md](BUILD.md) with full dependency and build instructions.

## Features

- GPU acceleration on NVIDIA CUDA GPUs.
- Massive parallel Hash160 search across many CUDA threads.
- Batch elliptic-curve operations and modular inversion.
- Configurable grid and batch sizing with `--grid`.
- Kernel slicing with `--slices` for better long-run behavior on high-end GPUs.
- Multi-GPU selection with `--gpus`.
- Direct P2PKH address target or raw Hash160 target.
- Partial Hash160 prefix candidate logging with `--partial`.
- Low VRAM usage compared with many GPU key search tools.

## Requirements

Recommended:

- NVIDIA GPU with CUDA support.
- Recent NVIDIA driver.
- CUDA Toolkit 13.x for RTX 50-series / compute 12.x and newer toolchains.

Windows:

- Visual Studio 2022 Build Tools with the C++ workload.
- CMake 3.24 or newer.
- CUDA Toolkit 13.x.

Linux:

- GCC/G++.
- Make.
- CMake, optional but recommended.
- CUDA Toolkit 13.x.

See [BUILD.md](BUILD.md) for full package installation notes.

## Build Quick Start

### Windows

From a PowerShell session with CUDA, CMake, and MSVC available:

```powershell
cmake -S . -B build-msvc -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-msvc --config Release --parallel
.\build-msvc\Release\CUDACyclone.exe --help
```

If `cl` is not available in normal PowerShell, use "x64 Native Tools Command Prompt for VS 2022".

### Linux

Using Make:

```bash
make clean
make -j"$(nproc)"
./CUDACyclone --help
```

Using CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --parallel
./build/CUDACyclone --help
```

### WSL2

Install the NVIDIA Windows driver with WSL CUDA support, then install CUDA Toolkit inside WSL. Verify:

```bash
nvidia-smi
nvcc --version
```

Then build using the Linux commands above.

## CUDA Architecture Selection

Pick the CUDA architecture that matches your GPU:

| GPU family | CMake architecture |
|------------|--------------------|
| RTX 30xx   | `86`               |
| RTX 40xx   | `89`               |
| Hopper/H100| `90`               |
| RTX 50xx / Blackwell | `120` or `121` |

Examples:

```powershell
cmake -S . -B build-msvc -A x64 -DCMAKE_CUDA_ARCHITECTURES=89
cmake -S . -B build-msvc -A x64 -DCMAKE_CUDA_ARCHITECTURES="89;120"
```

The Makefile queries `nvcc --list-gpu-arch` and automatically filters known architectures supported by the installed CUDA toolkit.

## Usage

General form:

```bash
./CUDACyclone --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [options]
```

Windows:

```powershell
.\CUDACyclone.exe --range <start_hex>:<end_hex> --address <base58> [options]
```

### Options

- `--range <start_hex>:<end_hex>`: Search range. The range should be a power-of-two-sized puzzle range.
- `--address <base58>`: Target compressed P2PKH Bitcoin address.
- `--target-hash160 <hash160_hex>`: Raw 20-byte Hash160 target as 40 hex characters.
- `--grid A,B`: Tuning parameter. `A` is points per thread batch, `B` is threads per batch/group.
- `--slices N`: Number of batches per thread per kernel launch.
- `--gpus GPU1,GPU2,...`: Select GPU IDs to use.
- `--random-interval SECONDS`: Re-randomize search segments periodically.
- `--partial HEX_DIGITS`: Save candidates whose Hash160 begins with the requested number of target hex digits.

Example:

```bash
./CUDACyclone --range FAC875:6FAC3875 --address 128z5d7nN7PkCuX5qoA4Ys6pmxUYnEy86k --partial 6
```

With `--partial 6`, a 6-hex-character Hash160 prefix match is saved to `partial.txt`. Longer matches are saved to `partialpN.txt`, where `N` is the number of extra matching hex digits.

## Tuning Notes

The original project notes that very large batches can reduce speed on some GPUs. For RTX 4090-class hardware, a commonly reported stable configuration is:

```bash
./CUDACyclone --range <start:end> --address <address> --grid 128,128 --slices 16
```

Good values are GPU-dependent. Start with the examples in this README, watch VRAM usage and speed, then adjust `--grid` and `--slices`.

## Proof Script

The included `proof.py` script helps verify that keys are not skipped. It generates random scalars in a range, calculates addresses, runs Cyclone, and reports how many test keys were found.

Usage:

```bash
python3 proof.py --range 200000000:3FFFFFFFF --grid 512,512
```

Typical successful summary:

```text
================ Summary by blocks ================
Range start A (start+2k)           : total= 128  success= 128  fail=   0
Range start B (start+1+2k)         : total= 128  success= 128  fail=   0
Range end A (end-2k)               : total= 128  success= 128  fail=   0
Range end B (end-1-2k)             : total= 128  success= 128  fail=   0
Full mod 512 residue coverage      : total= 256  success= 256  fail=   0
Random Q1 (0-25%)                  : total=  20  success=  20  fail=   0
Random Q2 (25-50%)                 : total=  20  success=  20  fail=   0
Random Q3 (50-75%)                 : total=  20  success=  20  fail=   0
Random Q4 (75-100%)                : total=  20  success=  20  fail=   0

Done. Results in cyclone_tests_results.txt. Successes=848 Failures=0
```

## Community Benchmarks

Reported speeds from the original README and community notes:

| GPU               | Grid      | Speed        | Notes            |
|-------------------|-----------|--------------|------------------|
| RTX 4090          | 128,1024  | 6214 Mkeys/s | Community report |
| RTX 4090          | 512,512   | 6038 Mkeys/s | Community report |
| RTX 4060          | 512,512   | 1238 Mkeys/s | Original report  |
| RTX 4070 Ti Super | 512,1024  | 3170 Mkeys/s | Community report |
| L4-2Q             | 512,256   | 1360 Mkeys/s | Community report |
| RTX 3070 mobile   | 256,256   | 1150 Mkeys/s | Community report |
| RTX 5090          | 128,256   | 8408 Mkeys/s | Original report  |

Benchmark results depend heavily on driver version, CUDA toolkit, clocks, thermals, grid settings, and puzzle range.

## Example Output

```text
======== PrePhase: GPU Information ====================
Device               : NVIDIA GeForce RTX 4060 (compute 8.9)
SM                   : 24
ThreadsPerBlock      : 256
Blocks               : 4096
Points batch size    : 512
Batches/SM           : 256
Memory utilization   : 6.9% (538.3 MB / 7.63 GB)
-------------------------------------------------------
Total threads        : 1048576

======== Phase-1: BruteForce ==========================
Time: 8.0 s | Speed: 1268.9 Mkeys/s | Count: 10204470016 | Progress: 7.42 %

======== FOUND MATCH! =================================
Private Key   : 00000000000000000000000000000000000000000000000000000022382FACD0
Public Key    : 03C060E1E3771CBECCB38E119C2414702F3F5181A89652538851D2E3886BDD70C6
```

## Build Artifacts

Generated binaries and build folders are intentionally ignored by git:

- `build/`
- `build-msvc/`
- `dist/`
- `*.o`, `*.obj`, `*.exe`
- root-level `CUDACyclone`

To package a release, build fresh binaries for the target platforms and place them in a release archive outside the tracked source files.

## Version Notes

Original CUDACyclone notes:

- `V1.3`: Full CUDA kernel rewrite again for preventing key skipping.
- `V1.2`: Full CUDA kernel rewrite.
- `V1.1`: Switch `pGx`/`pGy` to constant memory due to VRAM thermal throttling.
- `V1.0`: Release.

MultiCyclone fork notes:

- Native Windows + Linux build support.
- CUDA 13.x architecture updates.
- MSVC-compatible arithmetic changes.
- CMake build system.
- Build/package documentation.

Tips are highly recommended 
Solana: AMt9VC3Zuq98V7rXfTZ3MRgv8DXGRadGeYBmNwbSa95s
BTC: bc1q37vyyq7sjzkc3wx29c2ctgh7l25r5jy98xdkze
