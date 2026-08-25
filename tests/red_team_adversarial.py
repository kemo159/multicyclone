#!/usr/bin/env python3
"""Adversarial checks for candidate-to-HASH160 Build increments.

This is deliberately separate from the positive-vector correctness oracle.  It
adds negative controls that force the 32-bit prefix gate to fire without
allowing a false full match, plus byte/word-order target mutations.
"""

from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import sys
from pathlib import Path

from correctness_oracle import PUBKEY_VECTORS, check_reference_vectors

P = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
GX = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798
GY = 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8


def point_add(a: tuple[int, int] | None, b: tuple[int, int] | None):
    if a is None:
        return b
    if b is None:
        return a
    x1, y1 = a
    x2, y2 = b
    if x1 == x2 and (y1 + y2) % P == 0:
        return None
    slope = ((3 * x1 * x1) * pow(2 * y1, -1, P) if a == b else
             (y2 - y1) * pow(x2 - x1, -1, P)) % P
    x3 = (slope * slope - x1 - x2) % P
    return x3, (slope * (x1 - x3) - y1) % P


def compressed_pubkey(scalar: int) -> bytes:
    result, addend = None, (GX, GY)
    value = scalar
    while value:
        if value & 1:
            result = point_add(result, addend)
        addend = point_add(addend, addend)
        value >>= 1
    assert result is not None
    x, y = result
    return bytes((2 | (y & 1),)) + x.to_bytes(32, "big")


def run(executable: Path, target: str, scalar: int, timeout: int, grid: str = "2,32", extra_args: list[str] | None = None) -> str:
    # Keep the target scalar at the inclusive lower boundary.  A one-key
    # declared interval also prevents negative controls from becoming long
    # absent-target searches; --seconds bounds binaries that poll completion.
    search_range = f"{scalar:x}:{scalar + 1:x}"
    command = [
        str(executable), "--range", search_range, "--target-hash160", target,
        "--grid", grid, "--slices", "1", "--tpb", "32", "--seconds", "1",
    ]
    command.extend(extra_args or [])
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    output = result.stdout + result.stderr
    if result.returncode:
        raise AssertionError(f"GPU process failed ({result.returncode})\n{output}")
    return output


def mutations(hash160_hex: str) -> tuple[tuple[str, str], ...]:
    raw = bytes.fromhex(hash160_hex)
    same_prefix = raw[:4] + raw[4:-1] + bytes((raw[-1] ^ 1,))
    reversed_bytes = raw[::-1]
    reversed_words = b"".join(
        raw[offset:offset + 4][::-1] for offset in range(0, len(raw), 4)
    )
    return (
        ("same 32-bit prefix, different tail", same_prefix.hex()),
        ("all bytes reversed", reversed_bytes.hex()),
        ("bytes reversed within 32-bit words", reversed_words.hex()),
    )


def check_gpu(executable: Path, all_vectors: bool, timeout: int, grid: str, extra_args: list[str]) -> None:
    vectors = PUBKEY_VECTORS if all_vectors else PUBKEY_VECTORS[:1]
    for scalar, _, target, _ in vectors:
        positive = run(executable, target, scalar, timeout, grid, extra_args)
        if "FOUND MATCH" not in positive or f"{scalar:064X}" not in positive:
            raise AssertionError(f"positive control failed for scalar {scalar:x}\n{positive}")
        for label, mutated in mutations(target):
            negative = run(executable, mutated, scalar, timeout, grid, extra_args)
            if "FOUND MATCH" in negative:
                raise AssertionError(
                    f"false positive for scalar {scalar:x}: {label}, target={mutated}\n{negative}"
                )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path)
    parser.add_argument("--all-vectors", action="store_true")
    parser.add_argument("--fuzz", type=int, default=0,
                        help="run deterministic CPU-derived random positive GPU vectors")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--grid", default="2,32", help="GPU grid A,B used by adversarial tests")
    parser.add_argument("--extra-arg", action="append", default=[], help="extra executable argument (repeatable)")
    args = parser.parse_args()

    check_reference_vectors()
    for _, _, target, _ in PUBKEY_VECTORS:
        variants = mutations(target)
        assert variants[0][1][:8] == target[:8] and variants[0][1] != target
        assert all(len(value) == 40 for _, value in variants)
    print("Adversarial mutation construction: PASS")

    if args.executable:
        check_gpu(args.executable.resolve(), args.all_vectors, args.timeout, args.grid, args.extra_arg)
        count = len(PUBKEY_VECTORS) if args.all_vectors else 1
        print(f"GPU positive/negative adversarial checks: PASS ({count} vector(s))")
        rng = random.Random(0xC0DEC0DE)
        for _ in range(args.fuzz):
            scalar = rng.randrange(0x200000, 0x400000)
            digest = hashlib.new("ripemd160", hashlib.sha256(compressed_pubkey(scalar)).digest()).hexdigest()
            output = run(args.executable.resolve(), digest, scalar, args.timeout, args.grid, args.extra_arg)
            if "FOUND MATCH" not in output or f"{scalar:064X}" not in output:
                raise AssertionError(f"randomized positive failed for scalar {scalar:x}\n{output}")
        if args.fuzz:
            print(f"Deterministic randomized CPU-derived GPU vectors: PASS ({args.fuzz})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
