# Performance work

**Result: +17.3% over the v2.5 baseline**, measured on an RTX 5080 (sm_120) with
CUDA 13.1 at `--grid 512,512`.

Confirmed on both platforms, each measured against v2.5 built by the same
toolchain in the same environment (palindromic alternation, warm-up discarded):

| platform | v2.5 | optimized | gain |
|---|---|---|---|
| Windows (MSVC + CUDA 13.3) | 4276.9 | 5016.6 | **+17.3%** |
| Linux (make + CUDA 13.1, under WSL2) | 4229.1 | 4979.5 | **+17.7%** |

### A note on absolute numbers and search-range size

Throughput depends on how large a range each thread gets. On a wide range the
optimized build reads ~5000 Mkeys/s on both platforms. On a narrow 40-bit range
(`8000000000:FFFFFFFFFF`) the same binaries read ~4900 on Windows (−2%) and
~4230 under WSL2 (−15%): a short range means many more kernel launches and
host↔device transitions, and WSL2 pays a higher cost per transition.

So compare like with like. A number from a 40-bit hunt is not comparable to a
number from a wide-range benchmark, and a WSL2 number is not comparable to a
native one — but the *ratio* between two builds survives all of it, which is why
the table above is stated as ratios.

Everything below was measured, not estimated. Where a change turned out to be
worthless the measurement is recorded too, so nobody spends the effort twice.

---

## On Blackwell (sm_120), the CUDA toolkit is worth ~10%

Same source, same instruction count (SHA256 1384, HASH160 2080 instructions),
RTX 5080 / sm_120:

| toolkit | SHA256 | HASH160 |
|---|---|---|
| CUDA 12.8 | 9428 Mhash/s | 6225 Mhash/s |
| CUDA 13.0 | 9470 | 6052 |
| **CUDA 13.1** | **10178** | **6846** |
| CUDA 13.3 | neutral vs 13.1 (−0.67% / −0.33%, inside noise) | |

**+8% / +10% from the compiler alone.** The instruction *count* is identical
across all three — the difference is `ptxas` scheduling and instruction mix
(13.1 splits `IADD3`/`IADD` where the others emit `IADD3` throughout). A static
analysis would call these builds identical.

### This is architecture-specific — do not expect it on older GPUs

**CUDA 12.8 was the first toolkit to support `sm_120` at all**, so its Blackwell
backend was brand new and 13.1 had a year of maturing on it. What the table
above measures is a young code generator catching up, not a general compiler
improvement.

Confirmed in practice: on an **RTX 4090 (`sm_89`, supported since CUDA 11.8)**,
upgrading to 13.3 produced **no measurable change**. That backend was already
mature by 12.8, so there was nothing left to recover.

Rule of thumb: the newer the architecture relative to your toolkit, the more a
toolkit upgrade is likely to buy. On a mature target, expect nothing. Newer is
still never *worse*, so upgrading remains reasonable — just do not budget for a
10% win unless you are on a recently-released GPU.

---

## What changed in the code

| change | effect | verification |
|---|---|---|
| `RCFieldMul.cuh` field primitives (`MulModP` / `SqrModP` / `SubModP`) | bulk of the gain | 4.19M-vector bit-identical equivalence vs the original routines, including aliased call forms |
| `InvModP` divsteps modular inverse | +27.7% on the routine (~1.16% of runtime) | 4.19M random vectors over the full 256-bit range + 24 edge cases; bit-identical **and** `a·inv == 1` |
| `NegModP` | free | compiles to byte-identical SASS |
| batch size default 128 → 512 | +7.25% | alternated A/B |
| dead code removal (~150 lines) | hygiene | oracle + red team |
| GPU memory sizing fix | correctness | previously counted 2 of 6 device arrays and ignored the per-thread spill frame |
| Makefile: drop `CUDAHash.cu` from `SRC` | **fixes the Linux build** | see below |

### The Linux build fix

`CUDACyclone.cu` `#include`s `CUDAHash.cu` directly, so the hash routines inline
into the fused kernel instead of going through device linking. The Makefile also
compiled `CUDAHash.cu` as a second translation unit, which defined `K`, `IV` and
`verifyHash160_33_from_limbs_rare` twice — `nvlink` then failed with
`Multiple definition of ...`.

`CMakeLists.txt` only ever built `CUDACyclone.cu`, so **Windows never showed
this**. Keep the two build systems in step.

### Compile-time switches

All default ON, all independently A/B-able:

```
-DCUDACYCLONE_RC_FIELD_MUL=ON|OFF   # RC field primitives (~3.7% end-to-end)
-DCUDACYCLONE_RC_NEG=ON|OFF         # RC negation (identical SASS either way)
-DCUDACYCLONE_RC_INV=ON|OFF         # RC modular inverse
-DCUDACYCLONE_GNY_TABLE=ON|OFF      # precomputed -Gy table; OFF frees 16 KB constant
```

`RCFieldMul.cuh` is derived from RetiredCoder's GPLv3 work and carries a
provenance header. Build with `-DCUDACYCLONE_RC_FIELD_MUL=OFF` to exclude it.

---

## Things that look like wins and are not

Each of these was implemented or screened and then **measured**:

- **More occupancy is worse.** `KERNEL_MIN_BLOCKS` 3 and 4 cut registers to 80
  and 64 (768 / 1024 threads per SM) and measured **−1.11%** and **−2.13%**.
  Cutting registers is nearly free — the frame is dominated by the 8 KB `subp`
  array, so 123→80 registers costs only +96 bytes of stack — but the occupancy it
  buys is worthless because the kernel is ALU-throughput-bound. Extra warps add
  no throughput while multiplying local-memory traffic by the thread count.
- **`ptxas --register-usage-level` does nothing here.** All 11 values (0–10)
  produce identical code, because `__launch_bounds__` already binds the register
  cap.
- **Interleaving SHA256 chains cannot help.** Identical work spread over 1, 2 or
  4 independent dependency chains takes the same time (10120 / 10060 / 10050
  Mhash/s) — there is no idle issue slack to exploit.
- **PRMT byte-packing gains nothing.** Rewriting the 9-word limb→message packing
  as explicit `__byte_perm` produced **byte-identical SASS**. nvcc already
  canonicalises the shift/or form.
- **CompileIQ auto-tuning found nothing** in a 24-candidate search. The best
  candidate measured +0.98% during the search and **−0.48%** under a proper
  alternated A/B.

The recurring lesson: check what `nvcc` already did before hand-optimising. Three
separate times the compiler had already applied the transformation.

---

## Benchmarking this program honestly

The GPU drifts several percent with thermal state, and the program's own
`Speed:` gauge is a cumulative average that hides it.

1. **Alternate builds within one session**, palindromic order (A B B A). Never
   compare against a number from an earlier session — the absolute level moves
   ~1–3% between sessions while a same-session ratio reproduces to ~0.01%.
2. **Discard ~5 minutes of warm-up.** Throughput decays ~2.6% over the first few
   minutes of sustained load, then holds flat. There is no ramp *up*.
3. **Score with the median of per-second `Count` deltas**, skipping samples whose
   timestamp gap is not ~1.0 s — a logging desync makes every ~10th sample read
   ~9% low, and endpoint-differencing two `Count` values inherits it (measured
   0.9–1.4% low in every run).
4. **Treat anything under 0.5% as no change.** The steady-state noise floor is
   0.13%, verified by measuring three identical binaries.

A 60 s run scored this way resolves better than a 200 s run scored by endpoint
differencing.

---

## Verifying a build

```bash
python tests/correctness_oracle.py    --executable ./CUDACyclone --grid 512,512
python tests/red_team_adversarial.py  --executable ./CUDACyclone
```

Run these after **any** toolkit change. This code builds 256-bit carry chains
from separate `asm volatile` statements (`add.cc.u64` sets the carry, the next
`addc.cc.u64` consumes it). Nothing in the PTX contract stops a compiler from
scheduling a carry-clobbering instruction between them — it works because
`ptxas` keeps them adjacent. A compiler upgrade is exactly when that could
break, and it would break **silently**: wrong field arithmetic makes the search
quietly miss the target key rather than crash.

Verified passing on CUDA 13.1 and 13.3.

End-to-end check (recovers a known key in a 40-bit range, ~330 billion keys
scanned):

```bash
./CUDACyclone --range 8000000000:FFFFFFFFFF \
              --target-hash160 fd22d3a50b1cf94281f54a2a51279b4cb9133aee \
              --grid 512,512
# expected: Private Key ... D999999998
```

## Hash-counter flush granularity

`FLUSH_THRESHOLD` (now `HASH_FLUSH_THRESHOLD`, CUDACyclone.cu) was 65536, but a
thread only does `slices * B` keys per launch -- 32768 at the default `64 x 512`
-- so `MAYBE_WARP_FLUSH()` never fired and `hashes_accum` advanced *only* at
kernel end, in 5.6-billion-key lumps roughly every 1.2 s. The host samples the
rate once a second, so that aliased into a large swing in the displayed Mkeys/s
while actual throughput was steady.

Lowering it to 4096 makes the counter advance ~8x per launch. Measured over 30 s
runs on sm_120, samples after warm-up:

| config | jitter (sd) before | after | mean before | mean after |
|---|---|---|---|---|
| no CPU sidecar | 10.0% | 3.7% | 4853 | 4845 |
| `--cpu-threads 23` | 10.8% | 1.4% | 4763 | 4877 |

Throughput is unchanged -- the extra `atomicAdd` is one per warp per 4096
per-thread iterations, which does not register. Lowering `--slices` also smooths
the display (2.9% at `--slices 8`) but costs ~6% throughput (4569 vs 4853), so it
is not the right lever. Keep it a power of two: the test is a bitmask.
