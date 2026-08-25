#!/usr/bin/env python3
"""Independent fixed-vector oracle for the candidate-to-address pipeline."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path


SHA256_VECTORS = (
    (b"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    (b"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"),
    (b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"),
)

RIPEMD160_VECTORS = (
    (b"", "9c1185a5c5e9fc54612808977ee8f548b2258d31"),
    (b"a", "0bdc9d2d256b3ee9daae347be6f4dc835a467ffe"),
    (b"abc", "8eb208f7e05d987a9b044a8e98c6b087f15a0bfc"),
    (b"message digest", "5d0689ef49d2fae572b881b123a85ffa21595f36"),
)

# Frozen compressed secp256k1 public-key vectors in a practical GPU test range.
PUBKEY_VECTORS = (
    (0x100001, "039fb8987e3cb30a1174b7d64de26347166841051854696930396502714f6bcf4b", "dc53039e721d0d4315c37786b2db113dbaf2e49e", "1M5y78JuCQmnRAn2tc3Kuo72KkK8ecoTB6"),
    (0x100002, "037e24e7f1f7294eede3a19ba5c7d7e5f1a5b48d81a1f8dd3363f9306a4b526806", "3452b303685d398a239b03c42e8123595d0395b4", "15mfD2EUo1MBvkE81m6NNK3nQmUwJavZ1E"),
    (0x100003, "035839b5a8b09da19012f6ba0f949b0c685e617dae2a10ed38cdab8510e96c1c1b", "e81fb5ddaa7ce06bf08f37cb70398a92d30906a9", "1NAMknQQhdr3wqDvFBuiBMNpYT8aChryvp"),
    (0x100004, "03a01468c9be196a87b7e61058376e3cccfa301acdec50bf31e4df3d2007e55f48", "d398fc2da08526455ad289e028a80d8e7937086e", "1LHpr4nisLw8BPUfh8KjTQUnFzo7XRhD2K"),
    (0x100005, "0351ab97f560a3a9d655cd35d9101d507f425857f032ab58b74a885ffbd37fbc18", "2184b4e67746b083f402062640f1bdb4e5bc158d", "144EGx5AKNyG6vviSq9bY49oj5QZAsFznN"),
    (0x100006, "0237adaca237a1fc28322cc85e3c5e95a33923c23768f5f82ed943d778cca08816", "67a39c71b6425759d88f773916fe2ca8e12d31e3", "1ASzbf9h7ChBvSHukurEf5EyvLbJsfVRxv"),
    (0x100007, "0232182ce1be0371b02882f0258d8d25df93eac6c8e6f4e759e683c4565b0f26fc", "61ea004e9260a11f4c0b59539be5508e8da6cd89", "19visJAMhpgPPkiP6Reyed2Lvw8fHdyRn9"),
    (0x100008, "02c285ebbc248d94a72c20da0492d100cb50294ef84d1a6c4c187dcbf2eb013b21", "9148397b5c447b254b476190d6f88de0a43f047a", "1EFBW4jQ1yqo3pmZftN41qhDLrN9Yev35u"),
    (0x100009, "0387c79df5f6faf13de94ee76e3f9f32a7678caafc2dbe7a75ca9c4d39ddde7fb2", "9555fecf4e6fdff2be9c620ea201d194058b7ec4", "1EccheNwLu1tWSSGaa6BSfopjHStCYgVNr"),
    (0x10000A, "025b440ae11af9bf1441af8d299e8f9dacb86459543b98ddeeeb36ca2255f2e214", "06115f73bc9b1e2e3d5692f60c419a90dcabe1b5", "1Z5sEq7TuqpaKsCWconY96WvAv81aSu2k"),
)

BASE58 = b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


def p2pkh_address(hash160: bytes) -> str:
    payload = b"\x00" + hash160
    raw = payload + hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    zeros = len(raw) - len(raw.lstrip(b"\0"))
    n, encoded = int.from_bytes(raw, "big"), bytearray()
    while n:
        n, digit = divmod(n, 58)
        encoded.append(BASE58[digit])
    return (BASE58[:1] * zeros + bytes(reversed(encoded))).decode("ascii")


def check_reference_vectors() -> None:
    for message, expected in SHA256_VECTORS:
        assert hashlib.sha256(message).hexdigest() == expected
    for message, expected in RIPEMD160_VECTORS:
        assert hashlib.new("ripemd160", message).hexdigest() == expected
    for _, pubkey_hex, expected_hash, expected_address in PUBKEY_VECTORS:
        digest = hashlib.new("ripemd160", hashlib.sha256(bytes.fromhex(pubkey_hex)).digest()).digest()
        assert digest.hex() == expected_hash
        assert p2pkh_address(digest) == expected_address


def check_integrated_gpu(executable: Path, grid: str, extra_args: list[str]) -> None:
    for scalar, _, hash160_hex, _ in PUBKEY_VECTORS:
        command = [str(executable), "--range", "80000:180000", "--target-hash160", hash160_hex,
                   "--grid", grid, "--slices", "1", "--tpb", "32", *extra_args]
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        output = result.stdout + result.stderr
        if result.returncode or f"{scalar:064X}" not in output or "FOUND MATCH" not in output:
            raise AssertionError(f"integrated GPU vector {scalar} failed\n{output}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", type=Path, help="also test the integrated GPU pipeline")
    parser.add_argument("--grid", default="2,32", help="GPU grid A,B used by integrated tests")
    parser.add_argument("--extra-arg", action="append", default=[], help="extra executable argument (repeatable)")
    args = parser.parse_args()
    check_reference_vectors()
    print(f"Reference SHA-256/RIPEMD-160/HASH160 oracle: PASS ({len(PUBKEY_VECTORS)} pubkeys)")
    if args.executable:
        check_integrated_gpu(args.executable.resolve(), args.grid, args.extra_arg)
        print(f"Integrated GPU candidate pipeline: PASS ({len(PUBKEY_VECTORS)} scalars)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
