#!/usr/bin/env python3
"""End-to-end check that idle GPUs take over the CPU sidecar's leftover work.

The check has to be arranged so it cannot pass by accident. It derives a private
key with its own secp256k1 implementation (verified against the frozen vector in
correctness_oracle.py), then places that key at the very top of the range so it
lands inside the CPU sidecar's tail. The GPU's own share stops well short of it:
a GPU pass overshoots its assigned end by at most threadsTotal*B keys, which is
tens of millions, while the CPU tail here is billions. So if the key is found at
all, the GPUs must have swept the sidecar's leftovers.

    python tests/check_cpu_takeover.py [--exe PATH] [--gpu-seconds N]

Exits 0 on pass, 1 on fail.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time

P = 2**256 - 2**32 - 977
N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8
B58 = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def _add(p, q):
    if p is None:
        return q
    if q is None:
        return p
    (x1, y1), (x2, y2) = p, q
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    if p == q:
        lam = (3 * x1 * x1) * pow(2 * y1, P - 2, P) % P
    else:
        lam = (y2 - y1) * pow(x2 - x1, P - 2, P) % P
    x3 = (lam * lam - x1 - x2) % P
    return (x3, (lam * (x1 - x3) - y1) % P)


def _mul(k):
    r, acc = None, (GX, GY)
    while k:
        if k & 1:
            r = _add(r, acc)
        acc = _add(acc, acc)
        k >>= 1
    return r


def address_for(k: int) -> str:
    assert 0 < k < N
    x, y = _mul(k)
    pub = bytes([2 + (y & 1)]) + x.to_bytes(32, "big")
    h160 = hashlib.new("ripemd160", hashlib.sha256(pub).digest()).digest()
    payload = b"\x00" + h160
    raw = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    n, out = int.from_bytes(raw, "big"), bytearray()
    while n:
        n, d = divmod(n, 58)
        out.append(B58[d])
    pad = len(raw) - len(raw.lstrip(b"\0"))
    return (B58[:1] * pad + bytes(reversed(out))).decode()


def default_exe() -> str:
    for c in ("./CUDACyclone.exe", "./CUDACyclone",
              "bin-windows/CUDACyclone.exe", "bin-linux/CUDACyclone"):
        if os.path.exists(c):
            return c
    sys.exit("no CUDACyclone binary found; pass --exe PATH")


def run(exe, args, timeout):
    return subprocess.run([exe] + args, capture_output=True, text=True,
                          timeout=timeout).stdout


def measure_gpu_mkeys(exe) -> float:
    """Short solo run to size the real test against this machine's GPU."""
    out = run(exe, ["--range", "1000000000000000:1FFFFFFFFFFFFFFF",
                    "--address", "1M5y78JuCQmnRAn2tc3Kuo72KkK8ecoTB6",
                    "--seconds", "5"], timeout=180)
    speeds = [float(m) for m in re.findall(r"Speed:\s*([0-9.]+)\s*Mkeys/s", out)]
    if not speeds:
        sys.exit("could not measure GPU speed:\n" + out[-2000:])
    return max(speeds)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=None)
    ap.add_argument("--gpu-seconds", type=int, default=15,
                    help="how long the GPU's own share should take (default 15)")
    a = ap.parse_args()
    exe = a.exe or default_exe()

    # self-check the curve code against the project's own frozen vector
    assert address_for(0x100001) == "1M5y78JuCQmnRAn2tc3Kuo72KkK8ecoTB6"
    print("secp256k1 oracle self-check: OK")

    mkeys = measure_gpu_mkeys(exe)
    print(f"measured GPU: {mkeys:.0f} Mkeys/s")

    # Key sits at the top of the range, so it falls in the sidecar's tail.
    key = 0x10000000001
    cpu_percent = 1
    total = int(mkeys * 1e6 * a.gpu_seconds / (1 - cpu_percent / 100))
    start = key - total + 1
    if start <= 0:
        sys.exit("range would start below 1; lower --gpu-seconds")
    cpu_len = total * cpu_percent // 100

    rng = f"{start:X}:{key:X}"
    addr = address_for(key)
    print(f"range {rng}  ({total/1e9:.1f}B keys), CPU tail {cpu_len/1e9:.2f}B at {cpu_percent}%")
    print(f"target 0x{key:X} -> {addr} (sits inside the CPU tail)")
    print("running...")

    for f in ("found_key.txt", "cpu_worker.stats"):
        if os.path.exists(f):
            os.remove(f)

    t0 = time.time()
    out = run(exe, ["--range", rng, "--address", addr,
                    "--cpu-threads", "1", "--cpu-percent", str(cpu_percent)],
              timeout=3600)
    elapsed = time.time() - t0

    took_over = "taking over the CPU sidecar" in out
    found = f"{key:064X}" in out.upper()
    cpu_only = cpu_len / 5.9e6  # a single AVX2 thread runs ~5.9 Mkeys/s

    print()
    for line in out.splitlines():
        if re.search(r"taking over|FOUND MATCH|Private Key|NOT FOUND", line):
            print("   " + line.strip())
    print(f"\n   elapsed {elapsed:.1f}s (CPU alone would need ~{cpu_only:.0f}s for that tail)")

    ok = took_over and found
    print("\nPASS: idle GPUs took over the CPU remainder and found the key"
          if ok else
          "FAIL: " + ("takeover did not fire " if not took_over else "")
                   + ("key not found" if not found else ""))
    if not took_over:
        print("  (if the CPU finished first the split was already balanced -"
              " retry with a larger --gpu-seconds)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
