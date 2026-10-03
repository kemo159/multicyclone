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
            [--resume] [--checkpoint FILE] [--autosavetimer SECONDS]
```

`--grid A,B` sets *A* = keys per batch per thread, *B* = batches per SM.
**`--grid 512,512` is the tuned default** and what every number here was
measured at. `A=512` alone is worth +7.25% over the old default of 128.

### Flag reference

One of `--address` or `--target-hash160` is required, plus `--range`. Everything
else has a working default.

| Flag | Default | Explanation | Example |
|---|---|---|---|
| `--range START:END` | *(required)* | Inclusive hex interval to search, no `0x` prefix. | `--range 400000000:7FFFFFFFF` |
| `--address BASE58` | — | P2PKH target address. Mutually exclusive with `--target-hash160`. | `--address 1Abc...` |
| `--target-hash160 HEX` | — | Target as a 40-hex-digit HASH160 instead of an address. | `--target-hash160 62e907b1...` |
| `--grid A,B` | `512,512` | *A* = keys per batch per thread, *B* = batches per SM. Tuned once per GPU generation; see the `--grid` note above. | `--grid 512,512` |
| `--slices N` | `1` | Batches per kernel launch, as a multiple of `--grid`'s *A*. Raising it reduces launch overhead on large sweeps at the cost of less frequent progress updates and slower Ctrl+C response. | `--slices 8` |
| `--tpb N` | — | Threads per block; must be a multiple of 32 in `32..256`. Leave unset unless you have a measured reason to change it. | `--tpb 256` |
| `--max-launch-keys N` | `6000000000` | Caps keys per kernel launch, splitting a huge sweep into several launches so one call can't run long enough to trip a driver timeout or stall Ctrl+C. `0` disables the cap. | `--max-launch-keys 6000000000` |
| `--gpus LIST` | all GPUs | Comma-separated CUDA device indices to use. Omit to use every visible GPU. | `--gpus 0,1` |
| `--seconds N` | run to completion | Stop after *N* seconds (exit code 2) and write a checkpoint, even if the range isn't exhausted. | `--seconds 3600` |
| `--resume` | off | Continue from `cyclone_checkpoint.txt` (or `--checkpoint`'s file). Requires the exact same target, `--range`, `--grid`, `--slices`, `--tpb`, and GPU set as the run that wrote it. | `--resume` |
| `--checkpoint FILE` | `cyclone_checkpoint.txt` | Checkpoint path, for both writing and `--resume`. | `--checkpoint run1.ckpt` |
| `--checkpoint-pass PASS` | none | Passphrase mixed into the checkpoint's encryption key. Prefer the `CUDACYCLONE_CHECKPOINT_PASS` environment variable over this flag — the flag is visible in `ps`/Task Manager. | `--checkpoint-pass "correct horse"` |
| `--autosavetimer SECONDS` | off | Rewrite the checkpoint every *N* seconds while the search runs, so a crash or `kill -9` costs at most one interval. Ignored with `--random-interval`. | `--autosavetimer 300` |
| `--cpu-threads N\|auto` | off (no CPU sidecar) | Give the CPU a sidecar slice of the range (see below). `auto` uses all logical cores but one. | `--cpu-threads 24` |
| `--cpu-percent P\|auto` | `5` | Percent of the *total* range handed to the CPU sidecar. `auto`/`--cpu-auto` benchmarks both sides first and picks a split that finishes together. | `--cpu-percent auto` |
| `--cpu-auto` | off | Shorthand for `--cpu-percent auto` plus `--cpu-threads auto` when neither is set explicitly. | `--cpu-auto` |
| `--cpu-bench-seconds N` | `3` | How long the one-time GPU/CPU benchmark runs when `--cpu-auto`/`--cpu-percent auto` is used. | `--cpu-bench-seconds 5` |
| `--cpu-exe PATH` | this binary, re-invoked with `--cpu-worker` | Use a different executable as the CPU sidecar instead of the embedded worker. See the sidecar protocol below if you build your own. | `--cpu-exe cpu_avx2\Cyclone.exe` |
| `--random-interval SECONDS` | off (linear sweep) | Instead of sweeping the range in order, resample a fresh random sub-interval every *N* seconds. No linear progress, so `--resume`/`--autosavetimer` don't apply. GPU-only — see the sidecar protocol note below for why the CPU worker doesn't need this. | `--random-interval 30` |
| `--partial HEXDIGITS` | off | Also log *partial* HASH160 matches (first *N* hex digits, `1..40`) to `partial.txt`, not just exact matches. Useful for sanity-checking a search is actually computing real candidates. | `--partial 8` |

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

#### Autosaving while the run continues

Ctrl+C and `--seconds` both exit cleanly, so they get a chance to save. A crash,
a power cut, or `kill -9` does not — and on a long search that loses everything
since the run started. **`--autosavetimer SECONDS`** rewrites the same
checkpoint every *N* seconds while the search keeps going, capping what a hard
stop can cost you at one interval:

```
CUDACyclone --range AAAA:BBBB --address 1Abc... --autosavetimer 300
[autosave] checkpoint written to cyclone_checkpoint.txt at 12.04% (48318382080 keys checked)
```

Recovery is the ordinary `--resume` path — an autosaved file is the same format
the exit path writes, encrypted the same way, and resumable by the same command.

The save is cheap: it copies the per-thread counters and writes a small file,
without draining the GPU pipeline, so the search does not pause while it runs.
The counters are read while kernels are still running, which is deliberately
safe — they only ever count down, so a mid-flight read understates progress and
a resume repeats a little work rather than skipping any. The file is written to
`<checkpoint>.tmp` and renamed over the target, so a crash *during* an autosave
cannot truncate the previous good checkpoint.

Pick the interval to match what you are protecting against; `300` (5 min) is a
reasonable default for an overnight run. `--autosavetimer` is ignored in random
mode, which has no linear progress to record.

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

### Writing a custom `--cpu-exe` sidecar

`--cpu-exe PATH` lets you swap in a different CPU search binary instead of the
embedded worker — the same way `CUDACyclone.exe` invokes *itself* with
`--cpu-worker` when you don't pass `--cpu-exe` at all. Whatever binary you give
it is spawned once per run with a fixed slice of the range and must implement
this interface:

| Flag | Required | Explanation | Example |
|---|---|---|---|
| `-a BASE58` | yes | P2PKH target address for this run. | `-a 1Abc...` |
| `-r START:END` | yes | The sidecar's assigned sub-range in hex, inclusive. Set fresh by the parent process every run — see the note below. | `-r 400000000:7FFFFFFFF` |
| `-t N` | no | Thread count. Omit to use all logical cores. | `-t 24` |
| `--stats-file PATH` | no | Write progress as `KEY=value` lines (`CPU_THREADS`, `CPU_MKEYS`, `CPU_CHECKED`, `CPU_ELAPSED`, `CPU_PROGRESS`, `CPU_DONE`, `CPU_FOUND`) so the parent can poll it without parsing stdout. | `--stats-file cpu_worker.stats` |
| `--quiet` | no | Suppress the human-readable progress block; still writes `--stats-file` and `found_keys.txt`. | `--quiet` |
| `--bench-seconds N` | no | Run for exactly *N* seconds and report a throughput benchmark instead of searching for a match — used by `--cpu-auto`'s tuning pass. | `--bench-seconds 3` |

On a match, write one line to `found_keys.txt` in the working directory:
`<64-hex privkey> <66-hex compressed pubkey> <WIF> <address>`. The parent polls
that file's size rather than scraping stdout.

There is deliberately no random-search flag in this interface. `CUDACyclone.exe`
owns range assignment end to end: every time it gives the sidecar work, it's a
fresh `-r START:END` picked by the parent (including on the GPU-takeover and
`--random-interval` paths above). A custom sidecar never needs to pick its own
sub-range or reseed itself — it only ever searches exactly the interval it was
just handed, linearly, and reports back.

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

## Toolkit upgrades

Re-verify after **any** toolkit change. This code builds 256-bit carry chains
out of separate `asm volatile` statements (`add.cc.u64` sets the carry, the next
`addc.cc.u64` consumes it). Nothing in the PTX contract stops a compiler from
scheduling a carry-clobbering instruction between them — it works because
`ptxas` keeps them adjacent. A compiler upgrade is exactly when that could
break, and it would break **silently**: wrong field arithmetic means the search
quietly misses the target key rather than crashing.

Known good on CUDA 13.1 and 13.3.

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
