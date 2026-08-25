# CUDACyclone — optimized build

secp256k1 key search for CUDA. This tree is a performance-optimized fork of
CUDACyclone v2.5, itself derived from Dookoo2's Cyclone.

**Measured result: +17.3% over the v2.5 baseline** at `--grid 512,512` on an
RTX 5080 (sm_120), CUDA 13.1.

---

## Build it yourself — this matters more than the included binary

> **The CUDA toolkit version is worth more than any change in this repo.**
> Same source, same instruction count, measured on an RTX 5080:
>
> | toolkit | SHA256 | HASH160 |
> |---|---|---|
> | CUDA 12.8 | 9428 Mhash/s | 6225 Mhash/s |
> | CUDA 13.0 | 9470 | 6052 |
> | **CUDA 13.1+** | **10178** | **6846** |
>
> That is **+8% / +10% from the compiler alone**. Building with CUDA 12.8 gives
> up ~10% on the hashing half of the workload before you run a single line of
> this code. The difference is `ptxas` scheduling, not instruction count, so it
> is invisible to any static analysis. **Use the newest CUDA you can.**
> (13.3 measured neutral vs 13.1 — the jump is 12.8→13.1, not beyond.)

### Linux

```bash
make -j$(nproc)
./CUDACyclone --range <start_hex>:<end_hex> --address <base58> --grid 512,512
```

The Makefile auto-detects the architectures your `nvcc` supports and builds for
all of them. To build only for your own GPU (much faster compile):

```bash
make -j$(nproc) SM_ARCHS=120        # 120 = Blackwell / RTX 50-series
```

To pick a specific toolkit (recommended — see the warning above):

```bash
make -j$(nproc) CC=/usr/local/cuda-13.1/bin/nvcc SM_ARCHS=120
```

`bin-linux/CUDACyclone` is prebuilt with CUDA 13.1 on Ubuntu 24.04 for
**sm_120 only**. Verified: runs at ~4955 Mkeys/s and passes the oracle.

### Windows

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release --target CUDACyclone
```

If you have **CUDA 13.3+ installed**, CMake may fail with
`The CUDA Toolkit directory '' does not exist`. That is a Visual Studio
integration quirk, not a problem with this project. Add the toolset selector:

```powershell
-T cuda="C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.3"
```

`bin-windows/CUDACyclone.exe` is prebuilt with CUDA 13.3 for **sm_120 only**
(RTX 50-series). Any other GPU must build from source.

---

## Usage

```
CUDACyclone --range <start_hex>:<end_hex>
            (--address <base58> | --target-hash160 <40 hex digits>)
            [--grid A,B] [--slices N] [--tpb N] [--gpus 0,1] [--seconds N]
            [--resume] [--checkpoint FILE]
```

`--grid A,B` sets *A* = keys per batch per thread, *B* = batches per SM.
**`--grid 512,512` is the tuned default** and what every number here was
measured at. `A=512` alone is worth +7.25% over the old default of 128.

### Stopping and resuming

A run stopped with **Ctrl+C** or by **`--seconds`** writes its progress to
`cyclone_checkpoint.txt` (override with `--checkpoint FILE`). Re-run the *same
command* with `--resume` added to carry on from there:

```
CUDACyclone --grid 128,128 --slices 8 --range AAAA:BBBB --address 1Abc...
^C
Checkpoint saved to cyclone_checkpoint.txt at 18.32% (73282879488 keys checked)
Resume with the same command plus --resume

CUDACyclone --grid 128,128 --slices 8 --range AAAA:BBBB --address 1Abc... --resume
Resuming from cyclone_checkpoint.txt at 18.32% (73282879488 keys already checked)
```

The checkpoint is **encrypted**. It would otherwise spell out the address you are
hunting, the range, and how far you have got — exactly what you do not want
readable on a shared box, a backup, or a recovered disk. Only the file magic
stays in the clear:

```
CUDACycloneCheckpoint 1 enc
nonce=1f3c...        mac=9a41...        data=7be2...
```

The key is derived from the identity of the search — target hash160 plus range —
so resuming needs no extra secret: `--resume` already requires the same
`--address` and `--range`. Someone who has the file but does not know the target
cannot read it. Add `--checkpoint-pass PASS` (or set
`CUDACYCLONE_CHECKPOINT_PASS`) to mix in a passphrase as well, which also seals
it against someone who *does* know the target — but if you lose that passphrase
the progress is unrecoverable. The passphrase is visible in `ps`/Task Manager
when passed as an argument, so prefer the environment variable.

Cipher is SHA-256 in counter mode with encrypt-then-MAC over HMAC-SHA256, on a
fresh random nonce per write. The MAC is checked before anything is parsed, so a
truncated, edited or foreign checkpoint is rejected rather than half-read.

The checkpoint records the layout it was written under and `--resume` refuses to
run unless the new invocation reproduces it exactly — same **target**, same
**`--range`**, same **`--grid`**, same **`--slices`**, same **`--tpb`**, and the
same GPU set and per-GPU thread count. Each of those changes how the range is
tiled across threads, so a saved offset would no longer mean what it did. A
mismatch is reported field by field and exits non-zero rather than silently
searching the wrong keys:

```
Error: checkpoint 'cyclone_checkpoint.txt' does not match this run:
  - slices: checkpoint 8, now 16
```

Resuming is chainable — stop and resume as many times as you like. The
checkpoint is deleted once the range is finished or the key is found, so a
stale file can never restart a completed search. `--resume` is not supported
with `--random-interval` (random sweeps have no linear progress), and only GPU
progress is saved: with a `--cpu-threads` sidecar the CPU tail restarts from its
beginning, which the tool tells you when it saves.

### CPU sidecar: GPUs take over the leftovers

`--cpu-threads N` hands the CPU a `--cpu-percent` slice off the end of the range
(default 5%). That split is fixed up front, so if it over-allocates the CPU the
GPUs finish and then sit idle waiting. On this box the GPU runs ~4840 Mkeys/s and
all CPU cores together ~100 Mkeys/s, a ~48:1 ratio — the balanced CPU share is
about 2%, so the 5% default leaves the GPUs idling for well over half the run.

The GPUs no longer wait. When every GPU has finished its own share while the
sidecar is still going, the sidecar is stopped and the GPUs sweep what it had
left:

```
GPUs finished their share; taking over the CPU sidecar's remaining
FFCAD3992A - 10000000001 (0.89B keys)
```

Each CPU thread owns one contiguous chunk and walks it in order, so it reports
where it has got to and the takeover starts at the lowest such point. That is a
superset of the outstanding work — it re-covers what the *later* CPU threads had
already cleared — so nothing can be missed. The redundancy costs a fraction of a
second at GPU speed.

`--cpu-auto` still works and is worth using: it benchmarks both sides and picks
the split so they finish together, which avoids the wasted CPU effort in the
first place. The takeover is the safety net for when a one-time benchmark drifts
(thermal throttling, other load on the box) or when you set the percentage by hand.

### Exit codes

| code | meaning |
|---|---|
| 0 | key found, or range searched exhaustively without a match |
| 1 | bad arguments, or a `--resume` checkpoint that does not match |
| 2 | stopped by `--seconds` before the range was exhausted |
| 130 | interrupted with Ctrl+C |

Codes 2 and 130 mean the range was **not** fully searched — both leave a
checkpoint behind.

---

## What was changed

| change | effect | how it was verified |
|---|---|---|
| `RCFieldMul.cuh` field primitives (`MulModP`/`SqrModP`/`SubModP`) | bulk of the gain | 4.19M-vector bit-identical equivalence vs the original routines, incl. aliased call forms |
| `InvModP` divsteps modular inverse | +27.7% on the routine (~1.16% of runtime) | 4.19M random vectors + 24 edge cases; bit-identical **and** `a·inv == 1` |
| `NegModP` | free | compiles to byte-identical SASS |
| batch size default 128 → 512 | +7.25% | alternated A/B |
| dead code removal (~150 lines) | build hygiene | oracle + red team |
| GPU memory sizing fix | correctness | previously counted 2 of 6 arrays and ignored the spill frame |
| Makefile: drop `CUDAHash.cu` from `SRC` | fixes the Linux build | `CUDACyclone.cu` `#include`s it directly (so the hash path inlines into the fused kernel); compiling it as a second TU too defined `K`, `IV` and `verifyHash160_33_from_limbs_rare` twice and `nvlink` failed. CMake only ever built `CUDACyclone.cu`, so Windows never showed it. |

Compile-time switches (all default ON, all independently A/B-able):

```
-DCUDACYCLONE_RC_FIELD_MUL=ON|OFF   # RC field primitives
-DCUDACYCLONE_RC_NEG=ON|OFF         # RC negation
-DCUDACYCLONE_RC_INV=ON|OFF         # RC modular inverse
-DCUDACYCLONE_GNY_TABLE=ON|OFF      # precomputed -Gy table (OFF frees 16 KB constant)
```

### Licensing note

`RCFieldMul.cuh` is derived from RetiredCoder's GPLv3 work. That file carries a
provenance header. If GPLv3 is a problem for you, build with
`-DCUDACYCLONE_RC_FIELD_MUL=OFF` — everything still works, about 3.7% slower.

---

## Verifying a build

```bash
python tests/correctness_oracle.py    --executable ./CUDACyclone --grid 512,512
python tests/red_team_adversarial.py  --executable ./CUDACyclone
```

Run these after **any** toolkit change. This code builds 256-bit carry chains
out of separate `asm volatile` statements (`add.cc.u64` sets the carry, the next
`addc.cc.u64` consumes it). Nothing in the PTX contract stops a compiler from
scheduling a carry-clobbering instruction between them — it works because
`ptxas` keeps them adjacent. A compiler upgrade is exactly when that could
break, and it would break **silently**: wrong field arithmetic means the search
quietly misses the target key rather than crashing.

Both tests pass on CUDA 13.1 and 13.3.

---

## Tuning

`--grid 512,512` is the tuned optimum. Some things that look like wins and are
not, all measured rather than assumed:

- **More occupancy is worse.** `KERNEL_MIN_BLOCKS` 3 and 4 cut registers to 80
  and 64 (768/1024 threads per SM) and measured **−1.11%** and **−2.13%**. The
  kernel is ALU-throughput-bound, so extra warps add no throughput while
  multiplying local-memory traffic. Leave it at 2.
- **`ptxas --register-usage-level` does nothing here** — `__launch_bounds__`
  already binds the register cap; all 11 values give identical code.
- **Interleaving SHA256 chains does not help.** 1, 2 and 4 independent chains
  take the same time; there is no idle issue slack to exploit.
- **CompileIQ auto-tuning found no win** in a 24-candidate search — the best
  candidate measured +0.98% during the search and **−0.48%** under a proper
  alternated A/B.

## Benchmarking honestly

This GPU drifts several percent with thermal state, and the program's own
`Speed:` gauge is a cumulative average that hides it. To compare two builds:

1. Alternate them **in one session**, palindromic order (A B B A). Never compare
   against a number from an earlier session.
2. Discard ~5 minutes of warm-up — throughput decays ~2.6% over the first few
   minutes of sustained load, then holds flat.
3. Score with the **median of per-second `Count` deltas**, skipping samples whose
   timestamp gap isn't ~1.0 s (a logging desync makes every ~10th sample read
   ~9% low).
4. Treat anything under **0.5%** as no change.

Ratios from alternated runs reproduce to ~0.01%; absolute numbers are worthless
across sessions.
