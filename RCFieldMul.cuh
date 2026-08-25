// secp256k1 field multiplication for SM 8.9 / 12.0.
//
// ---------------------------------------------------------------------------
// PROVENANCE AND LICENSE -- READ BEFORE DISTRIBUTING
//
// The routines below are derived from RCKangaroo / RCKangTurbo:
//     (c) 2026, RetiredCoder (RC)
//     License: GPLv3  --  https://github.com/RetiredC
//
// GPLv3 is copyleft. Distributing a binary or source tree that includes this
// file makes the whole work subject to GPLv3, which requires releasing the
// corresponding source under the same terms. Building it privately creates no
// obligation. To remove the entanglement, build with
// -DCUDACYCLONE_RC_FIELD_MUL=OFF, which restores the original _ModMult and
// costs about 5% throughput; nothing else in the tree depends on this header.
// ---------------------------------------------------------------------------
//
// Why it is faster than the previous _ModMult (measured on an RTX 5080):
//   _ModMult, native 64x64 reduction ......... 52.29 Gmul/s
//   _ModMult, original shift/add reduction ... 48.61 Gmul/s
//   MulModP (this file) ...................... 63.51 Gmul/s   (+21%)
// The gain is not the reduction technique -- an independently written
// 32-bit-granular reduction measured 52.50, i.e. no better than nvcc's own
// code for the 64x64 multiply. It comes from carrying the entire 512-bit
// product and the mod-P fold on explicit add.cc/addc.cc carry chains, so no
// compiler-synthesised carry handling appears between the multiplies.
//
// Verified bit-exact against the previous _ModMult on 524288 random operand
// pairs, and against the agreement gate in the build.

#pragma once
#include <cstdint>

namespace rcfield {

#define RCF_add_cc_64(res, a, b)   asm volatile ("add.cc.u64 %0, %1, %2;"  : "=l"(res) : "l"(a), "l"(b))
#define RCF_addc_cc_64(res, a, b)  asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b))
#define RCF_addc_64(res, a, b)     asm volatile ("addc.u64 %0, %1, %2;"    : "=l"(res) : "l"(a), "l"(b))
#define RCF_add_cc_32(res, a, b)   asm volatile ("add.cc.u32 %0, %1, %2;"  : "=r"(res) : "r"(a), "r"(b))
#define RCF_addc_cc_32(res, a, b)  asm volatile ("addc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b))
#define RCF_addc_32(res, a, b)     asm volatile ("addc.u32 %0, %1, %2;"    : "=r"(res) : "r"(a), "r"(b))
#define RCF_mul_wide_32(res, a, b) asm volatile ("mul.wide.u32 %0, %1, %2;" : "=l"(res) : "r"(a), "r"(b))

// 2^256 == 2^32 + 977 (mod p), so the fold constant's small half is 977.
#define RCF_P_INV32 0x000003D1u


#define RCF_sub_cc_64(res, a, b)   asm volatile ("sub.cc.u64 %0, %1, %2;"  : "=l"(res) : "l"(a), "l"(b))
#define RCF_subc_cc_64(res, a, b)  asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(res) : "l"(a), "l"(b))
#define RCF_subc_64(res, a, b)     asm volatile ("subc.u64 %0, %1, %2;"    : "=l"(res) : "l"(a), "l"(b))
#define RCF_subc_32(res, a, b)     asm volatile ("subc.u32 %0, %1, %2;"    : "=r"(res) : "r"(a), "r"(b))
#define RCF_sub_cc_32(res, a, b)   asm volatile ("sub.cc.u32 %0, %1, %2;"  : "=r"(res) : "r"(a), "r"(b))
#define RCF_subc_cc_32(res, a, b)  asm volatile ("subc.cc.u32 %0, %1, %2;" : "=r"(res) : "r"(a), "r"(b))
#define RCF_madc_lo_32(res, a, b, c) \
    asm volatile ("madc.lo.u32 %0, %1, %2, %3;" : "=r"(res) : "r"(a), "r"(b), "r"(c))
#define RCF_P_0    0xFFFFFFFEFFFFFC2Full
#define RCF_P_123  0xFFFFFFFFFFFFFFFFull

// res = val1 - val2 mod p. Alias-safe: output word i depends only on input
// word i plus the borrow chain.
__device__ __forceinline__ void SubModP(uint64_t* res, const uint64_t* val1, const uint64_t* val2)
{
    RCF_sub_cc_64 (res[0], val1[0], val2[0]);
    RCF_subc_cc_64(res[1], val1[1], val2[1]);
    RCF_subc_cc_64(res[2], val1[2], val2[2]);
    RCF_subc_cc_64(res[3], val1[3], val2[3]);
    uint32_t carry;
    RCF_subc_32(carry, 0u, 0u);
    if (carry) {
        RCF_add_cc_64 (res[0], res[0], RCF_P_0);
        RCF_addc_cc_64(res[1], res[1], RCF_P_123);
        RCF_addc_cc_64(res[2], res[2], RCF_P_123);
        RCF_addc_64   (res[3], res[3], RCF_P_123);
    }
}

// res[0..9] (uint32_t words) = val256 * val64
// res = -res mod p (in place). Bit-identical to ModNeg256 for every non-zero
// input; -0 would give p rather than 0, which cannot occur for a curve point's y.
__device__ __forceinline__ void NegModP(uint64_t* res)
{
    RCF_sub_cc_64 (res[0], RCF_P_0,   res[0]);
    RCF_subc_cc_64(res[1], RCF_P_123, res[1]);
    RCF_subc_cc_64(res[2], RCF_P_123, res[2]);
    RCF_subc_64   (res[3], RCF_P_123, res[3]);
}

__device__ __forceinline__ void mul_256_by_64(uint64_t* res, const uint64_t* val256, uint64_t val64)
{
    uint64_t tmp64[7];
    uint32_t* tmp = (uint32_t*)tmp64;
    uint32_t* rs  = (uint32_t*)res;
    const uint32_t* a = (const uint32_t*)val256;
    const uint32_t* b = (const uint32_t*)&val64;

    RCF_mul_wide_32(res[0],   a[0], b[0]);
    RCF_mul_wide_32(tmp64[0], a[1], b[0]);
    RCF_mul_wide_32(tmp64[1], a[2], b[0]);
    RCF_mul_wide_32(tmp64[2], a[3], b[0]);
    RCF_mul_wide_32(tmp64[3], a[4], b[0]);
    RCF_mul_wide_32(tmp64[4], a[5], b[0]);
    RCF_mul_wide_32(tmp64[5], a[6], b[0]);
    RCF_mul_wide_32(tmp64[6], a[7], b[0]);

    RCF_add_cc_32 (rs[1], rs[1], tmp[0]);
    RCF_addc_cc_32(rs[2], tmp[1],  tmp[2]);
    RCF_addc_cc_32(rs[3], tmp[3],  tmp[4]);
    RCF_addc_cc_32(rs[4], tmp[5],  tmp[6]);
    RCF_addc_cc_32(rs[5], tmp[7],  tmp[8]);
    RCF_addc_cc_32(rs[6], tmp[9],  tmp[10]);
    RCF_addc_cc_32(rs[7], tmp[11], tmp[12]);
    RCF_addc_32   (rs[8], tmp[13], 0u);   // 8+1=9 words, so rs[9] cannot carry

    uint64_t kk[7];
    uint32_t* k = (uint32_t*)kk;
    RCF_mul_wide_32(kk[0],    a[0], b[1]);
    RCF_mul_wide_32(tmp64[0], a[1], b[1]);
    RCF_mul_wide_32(tmp64[1], a[2], b[1]);
    RCF_mul_wide_32(tmp64[2], a[3], b[1]);
    RCF_mul_wide_32(tmp64[3], a[4], b[1]);
    RCF_mul_wide_32(tmp64[4], a[5], b[1]);
    RCF_mul_wide_32(tmp64[5], a[6], b[1]);
    RCF_mul_wide_32(tmp64[6], a[7], b[1]);

    RCF_add_cc_32 (k[1], k[1],   tmp[0]);
    RCF_addc_cc_32(k[2], tmp[1],  tmp[2]);
    RCF_addc_cc_32(k[3], tmp[3],  tmp[4]);
    RCF_addc_cc_32(k[4], tmp[5],  tmp[6]);
    RCF_addc_cc_32(k[5], tmp[7],  tmp[8]);
    RCF_addc_cc_32(k[6], tmp[9],  tmp[10]);
    RCF_addc_cc_32(k[7], tmp[11], tmp[12]);
    RCF_addc_32   (k[8], tmp[13], 0u);

    RCF_add_cc_32 (rs[1], rs[1], k[0]);
    RCF_addc_cc_32(rs[2], rs[2], k[1]);
    RCF_addc_cc_32(rs[3], rs[3], k[2]);
    RCF_addc_cc_32(rs[4], rs[4], k[3]);
    RCF_addc_cc_32(rs[5], rs[5], k[4]);
    RCF_addc_cc_32(rs[6], rs[6], k[5]);
    RCF_addc_cc_32(rs[7], rs[7], k[6]);
    RCF_addc_cc_32(rs[8], rs[8], k[7]);
    RCF_addc_32   (rs[9], k[8],  0u);
}

__device__ __forceinline__ void add_320_to_256(uint64_t* res, const uint64_t* val)
{
    RCF_add_cc_64 (res[0], res[0], val[0]);
    RCF_addc_cc_64(res[1], res[1], val[1]);
    RCF_addc_cc_64(res[2], res[2], val[2]);
    RCF_addc_cc_64(res[3], res[3], val[3]);
    RCF_addc_64   (res[4], val[4], 0ull);
}

// res[0..9] (uint32_t) = val[0..7] (uint32_t) * (2^32 + 977):  the x977 partial products
// plus the one-word shift, both on 32-bit carry chains.
__device__ __forceinline__ void mul_256_by_P0inv(uint32_t* res, const uint32_t* val)
{
    uint64_t tmp64[7];
    uint32_t* tmp = (uint32_t*)tmp64;
    RCF_mul_wide_32(*(uint64_t*)res, val[0], RCF_P_INV32);
    RCF_mul_wide_32(tmp64[0], val[1], RCF_P_INV32);
    RCF_mul_wide_32(tmp64[1], val[2], RCF_P_INV32);
    RCF_mul_wide_32(tmp64[2], val[3], RCF_P_INV32);
    RCF_mul_wide_32(tmp64[3], val[4], RCF_P_INV32);
    RCF_mul_wide_32(tmp64[4], val[5], RCF_P_INV32);
    RCF_mul_wide_32(tmp64[5], val[6], RCF_P_INV32);
    RCF_mul_wide_32(tmp64[6], val[7], RCF_P_INV32);

    RCF_add_cc_32 (res[1], res[1], tmp[0]);
    RCF_addc_cc_32(res[2], tmp[1],  tmp[2]);
    RCF_addc_cc_32(res[3], tmp[3],  tmp[4]);
    RCF_addc_cc_32(res[4], tmp[5],  tmp[6]);
    RCF_addc_cc_32(res[5], tmp[7],  tmp[8]);
    RCF_addc_cc_32(res[6], tmp[9],  tmp[10]);
    RCF_addc_cc_32(res[7], tmp[11], tmp[12]);
    RCF_addc_32   (res[8], tmp[13], 0u);   // tmp[13] cannot be UINT_MAX, so res[9] cannot carry

    RCF_add_cc_32 (res[1], res[1], val[0]);
    RCF_addc_cc_32(res[2], res[2], val[1]);
    RCF_addc_cc_32(res[3], res[3], val[2]);
    RCF_addc_cc_32(res[4], res[4], val[3]);
    RCF_addc_cc_32(res[5], res[5], val[4]);
    RCF_addc_cc_32(res[6], res[6], val[5]);
    RCF_addc_cc_32(res[7], res[7], val[6]);
    RCF_addc_cc_32(res[8], res[8], val[7]);
    RCF_addc_32   (res[9], 0u, 0u);
}

// res[0..3] = val1 * val2 mod p. Leaves the same non-canonical residues the
// previous _ModMult did (verified bit-identical), so downstream comparisons
// against p+1 style values keep working.
__device__ __forceinline__ void MulModP(uint64_t* res, const uint64_t* val1, const uint64_t* val2)
{
    uint64_t buff[8], tmp[5], tmp2[2], tmp3;
    // full 512-bit product
    mul_256_by_64(tmp,  val1, val2[1]);
    mul_256_by_64(buff, val1, val2[0]);
    add_320_to_256(buff + 1, tmp);
    mul_256_by_64(tmp,  val1, val2[2]);
    add_320_to_256(buff + 2, tmp);
    mul_256_by_64(tmp,  val1, val2[3]);
    add_320_to_256(buff + 3, tmp);
    // fold the high 256 bits back in
    mul_256_by_P0inv((uint32_t*)tmp, (const uint32_t*)(buff + 4));
    RCF_add_cc_64 (buff[0], buff[0], tmp[0]);
    RCF_addc_cc_64(buff[1], buff[1], tmp[1]);
    RCF_addc_cc_64(buff[2], buff[2], tmp[2]);
    RCF_addc_cc_64(buff[3], buff[3], tmp[3]);
    RCF_addc_64   (tmp[4],  tmp[4],  0ull);
    // second, much smaller fold of what the first one carried out
    uint32_t* t32 = (uint32_t*)tmp;
    uint32_t* a32 = (uint32_t*)tmp2;
    uint32_t* kw  = (uint32_t*)&tmp3;
    RCF_mul_wide_32(tmp2[0], t32[8], RCF_P_INV32);
    RCF_mul_wide_32(tmp3,    t32[9], RCF_P_INV32);
    RCF_add_cc_32 (a32[1], a32[1], kw[0]);
    RCF_addc_32   (a32[2], kw[1],  0u);        // a32[3] cannot carry
    RCF_add_cc_32 (a32[1], a32[1], t32[8]);
    RCF_addc_cc_32(a32[2], a32[2], t32[9]);
    RCF_addc_32   (a32[3], 0u, 0u);

    RCF_add_cc_64 (res[0], buff[0], tmp2[0]);
    RCF_addc_cc_64(res[1], buff[1], tmp2[1]);
    RCF_addc_cc_64(res[2], buff[2], 0ull);
    RCF_addc_64   (res[3], buff[3], 0ull);
}

__device__ __forceinline__ void add_320_to_256s(uint32_t* res,
        uint64_t _v1, uint64_t _v2, uint64_t _v3, uint64_t _v4,
        uint64_t _v5, uint64_t _v6, uint64_t _v7, uint64_t _v8)
{
    uint32_t* v1 = (uint32_t*)&_v1; uint32_t* v2 = (uint32_t*)&_v2;
    uint32_t* v3 = (uint32_t*)&_v3; uint32_t* v4 = (uint32_t*)&_v4;
    uint32_t* v5 = (uint32_t*)&_v5; uint32_t* v6 = (uint32_t*)&_v6;
    uint32_t* v7 = (uint32_t*)&_v7; uint32_t* v8 = (uint32_t*)&_v8;

    RCF_add_cc_32 (res[0], res[0], v1[0]);
    RCF_addc_cc_32(res[1], res[1], v1[1]);
    RCF_addc_cc_32(res[2], res[2], v3[0]);
    RCF_addc_cc_32(res[3], res[3], v3[1]);
    RCF_addc_cc_32(res[4], res[4], v5[0]);
    RCF_addc_cc_32(res[5], res[5], v5[1]);
    RCF_addc_cc_32(res[6], res[6], v7[0]);
    RCF_addc_cc_32(res[7], res[7], v7[1]);
    RCF_addc_32   (res[8], res[8], 0u);

    RCF_add_cc_32 (res[1], res[1], v2[0]);
    RCF_addc_cc_32(res[2], res[2], v2[1]);
    RCF_addc_cc_32(res[3], res[3], v4[0]);
    RCF_addc_cc_32(res[4], res[4], v4[1]);
    RCF_addc_cc_32(res[5], res[5], v6[0]);
    RCF_addc_cc_32(res[6], res[6], v6[1]);
    RCF_addc_cc_32(res[7], res[7], v8[0]);
    RCF_addc_cc_32(res[8], res[8], v8[1]);
    RCF_addc_32   (res[9], 0u, 0u);
}

// Dedicated squaring: ~7% over a general multiply of a value by itself.
__device__ __forceinline__ void SqrModP(uint64_t* res, uint64_t* val)
{
	uint64_t buff[8], tmp[5], tmp2[2], tmp3, mm;
	uint32_t* a = (uint32_t*)val;
	uint64_t mar[28];
	uint32_t* b32 = (uint32_t*)buff;
	uint32_t* m32 = (uint32_t*)mar;
//calc 512 bits
	RCF_mul_wide_32(mar[0], a[1], a[0]); //ab
	RCF_mul_wide_32(mar[1], a[2], a[0]); //ac
	RCF_mul_wide_32(mar[2], a[3], a[0]); //ad
	RCF_mul_wide_32(mar[3], a[4], a[0]); //ae
	RCF_mul_wide_32(mar[4], a[5], a[0]); //af
	RCF_mul_wide_32(mar[5], a[6], a[0]); //ag
	RCF_mul_wide_32(mar[6], a[7], a[0]); //ah
	RCF_mul_wide_32(mar[7], a[2], a[1]); //bc
	RCF_mul_wide_32(mar[8], a[3], a[1]); //bd
	RCF_mul_wide_32(mar[9], a[4], a[1]); //be
	RCF_mul_wide_32(mar[10], a[5], a[1]); //bf
	RCF_mul_wide_32(mar[11], a[6], a[1]); //bg
	RCF_mul_wide_32(mar[12], a[7], a[1]); //bh
	RCF_mul_wide_32(mar[13], a[3], a[2]); //cd
	RCF_mul_wide_32(mar[14], a[4], a[2]); //ce
	RCF_mul_wide_32(mar[15], a[5], a[2]); //cf
	RCF_mul_wide_32(mar[16], a[6], a[2]); //cg
	RCF_mul_wide_32(mar[17], a[7], a[2]); //ch
	RCF_mul_wide_32(mar[18], a[4], a[3]); //de
	RCF_mul_wide_32(mar[19], a[5], a[3]); //df
	RCF_mul_wide_32(mar[20], a[6], a[3]); //dg
	RCF_mul_wide_32(mar[21], a[7], a[3]); //dh
	RCF_mul_wide_32(mar[22], a[5], a[4]); //ef
	RCF_mul_wide_32(mar[23], a[6], a[4]); //eg
	RCF_mul_wide_32(mar[24], a[7], a[4]); //eh
	RCF_mul_wide_32(mar[25], a[6], a[5]); //fg
	RCF_mul_wide_32(mar[26], a[7], a[5]); //fh
	RCF_mul_wide_32(mar[27], a[7], a[6]); //gh
//a
	RCF_mul_wide_32(buff[0], a[0], a[0]); //aa
	RCF_add_cc_32(b32[1], b32[1], m32[0]);
	RCF_addc_cc_32(b32[2], m32[1], m32[2]);
	RCF_addc_cc_32(b32[3], m32[3], m32[4]);
	RCF_addc_cc_32(b32[4], m32[5], m32[6]);
	RCF_addc_cc_32(b32[5], m32[7], m32[8]);
	RCF_addc_cc_32(b32[6], m32[9], m32[10]);
	RCF_addc_cc_32(b32[7], m32[11], m32[12]);
	RCF_addc_cc_32(b32[8], m32[13], 0);
	b32[9] = 0;
//b+	 
	RCF_mul_wide_32(mm, a[1], a[1]); //bb
	add_320_to_256s(b32 + 1, mar[0], mm, mar[7], mar[8], mar[9], mar[10], mar[11], mar[12]);
	RCF_mul_wide_32(mm, a[2], a[2]); //cc
	add_320_to_256s(b32 + 2, mar[1], mar[7], mm, mar[13], mar[14], mar[15], mar[16], mar[17]);
	RCF_mul_wide_32(mm, a[3], a[3]); //dd
	add_320_to_256s(b32 + 3, mar[2], mar[8], mar[13], mm, mar[18], mar[19], mar[20], mar[21]);
	RCF_mul_wide_32(mm, a[4], a[4]); //ee
	add_320_to_256s(b32 + 4, mar[3], mar[9], mar[14], mar[18], mm, mar[22], mar[23], mar[24]);
	RCF_mul_wide_32(mm, a[5], a[5]); //ff
	add_320_to_256s(b32 + 5, mar[4], mar[10], mar[15], mar[19], mar[22], mm, mar[25], mar[26]);
	RCF_mul_wide_32(mm, a[6], a[6]); //gg
	add_320_to_256s(b32 + 6, mar[5], mar[11], mar[16], mar[20], mar[23], mar[25], mm, mar[27]);
	RCF_mul_wide_32(mm, a[7], a[7]); //hh
	add_320_to_256s(b32 + 7, mar[6], mar[12], mar[17], mar[21], mar[24], mar[26], mar[27], mm);
//fast mod P
	mul_256_by_P0inv((uint32_t*)tmp, (uint32_t*)(buff + 4));
	RCF_add_cc_64(buff[0], buff[0], tmp[0]);
	RCF_addc_cc_64(buff[1], buff[1], tmp[1]);
	RCF_addc_cc_64(buff[2], buff[2], tmp[2]);
	RCF_addc_cc_64(buff[3], buff[3], tmp[3]);
	RCF_addc_64(tmp[4], tmp[4], 0ull);
//see mul_256_by_P0inv for details
	uint32_t* t32 = (uint32_t*)tmp;
	uint32_t* a32 = (uint32_t*)tmp2;
	uint32_t* k = (uint32_t*)&tmp3;
	RCF_mul_wide_32(tmp2[0], t32[8], RCF_P_INV32);
	RCF_mul_wide_32(tmp3, t32[9], RCF_P_INV32);
	RCF_add_cc_32(a32[1], a32[1], k[0]);
	RCF_addc_32(a32[2], k[1], 0); //we cannot get carry here for a32[3]
	RCF_add_cc_32(a32[1], a32[1], t32[8]);
	RCF_addc_cc_32(a32[2], a32[2], t32[9]);
	RCF_addc_32(a32[3], 0, 0);

	RCF_add_cc_64(res[0], buff[0], tmp2[0]);
	RCF_addc_cc_64(res[1], buff[1], tmp2[1]);
	RCF_addc_cc_64(res[2], buff[2], 0ull);
	RCF_addc_64(res[3], buff[3], 0ull);
}

// ---------------------------------------------------------------------------
// Modular inverse via the "divsteps" / half-delta algorithm of Bernstein-Yang,
// as implemented by RetiredCoder. Reference:
//   https://tches.iacr.org/index.php/TCHES/article/download/8298/7648/4494
//
// Replaces the in-tree DRS62 _ModInv (1128 SASS instructions). Operates on a
// 288-bit signed representation held as 9 x u32, so `res` must have room for
// 9 u32 words -- the callers pass a uint64_t[5] (320 bits), which satisfies it.
// res[8] is used as a sign word and is normalised to 0 before returning.
//
// Speed, measured (<scratchpad>/invbench.cu, 9 alternated rounds in one
// process, 13.1M inversions each): DRS62 843.3 Minv/s vs divsteps 1076.9
// Minv/s, i.e. **+27.7% on the routine**. Note this version is 80 instructions
// LARGER in static SASS (1208 vs 1128) and still faster -- it wins on loop trip
// count, not code size, so a static census is NOT a valid proxy when comparing
// two different algorithms (it is only valid when control flow is identical).
//
// Share of total runtime: the kernel needs 5e9/512 = 9.77M inversions/s and
// inversions alone sustain 843.3M/s, so they occupy ~1.16% of GPU time, which
// predicts ~+0.25% overall.
//
// That prediction did NOT show up end-to-end. An 8-run alternated A/B measured
// -0.19%, but the two runs driving that were also the only two with ~3x the
// IQR of the rest (disturbed runs); excluding them leaves -0.06%, i.e. nothing.
// The honest statement is that the end-to-end effect is somewhere around
// -0.1%..+0.2% and is not resolvable at this noise floor.
//
// Kept anyway, because there is no mechanism for harm and the routine is
// unambiguously faster: REG:126 is unchanged and the hot kernel's stack frame
// is 144 bytes SMALLER (8560 -> 8416). The likely reason the routine win does
// not translate is that at this occupancy the inversion phase overlaps other
// warps' hashing, so its marginal cost is below its standalone throughput --
// meaning a share computed from standalone throughput is an UPPER bound on the
// achievable end-to-end gain, not an estimate of it.
//
// Correctness is established by equivalence, not by the timing above:
// <scratchpad>/inv2.cu checks bit-identical output plus a*inv==1 against the
// builtin _ModMult over 4,194,304 random vectors spanning the full 256-bit
// range, plus 24 hand-picked edge vectors (1, 2, p-1, p-2, 2^255, every 2^k
// limb boundary, the reduction constant). All pass.
// ---------------------------------------------------------------------------
#ifndef CUDACYCLONE_RC_INV
#define CUDACYCLONE_RC_INV 1
#endif
#if CUDACYCLONE_RC_INV

__device__ __forceinline__ void inv_add_288(uint32_t* res, const uint32_t* val1, const uint32_t* val2)
{
	RCF_add_cc_32(res[0], val1[0], val2[0]);
	RCF_addc_cc_32(res[1], val1[1], val2[1]);
	RCF_addc_cc_32(res[2], val1[2], val2[2]);
	RCF_addc_cc_32(res[3], val1[3], val2[3]);
	RCF_addc_cc_32(res[4], val1[4], val2[4]);
	RCF_addc_cc_32(res[5], val1[5], val2[5]);
	RCF_addc_cc_32(res[6], val1[6], val2[6]);
	RCF_addc_cc_32(res[7], val1[7], val2[7]);
	RCF_addc_32(res[8], val1[8], val2[8]);
}

__device__ __forceinline__ void inv_neg_288(uint32_t* res)
{
	RCF_sub_cc_32(res[0], 0, res[0]);
	RCF_subc_cc_32(res[1], 0, res[1]);
	RCF_subc_cc_32(res[2], 0, res[2]);
	RCF_subc_cc_32(res[3], 0, res[3]);
	RCF_subc_cc_32(res[4], 0, res[4]);
	RCF_subc_cc_32(res[5], 0, res[5]);
	RCF_subc_cc_32(res[6], 0, res[6]);
	RCF_subc_cc_32(res[7], 0, res[7]);
	RCF_subc_32(res[8], 0, res[8]);
}

__device__ __forceinline__ void inv_mul_288_by_i32(uint32_t* res, const uint32_t* val288, int ival32)
{
	uint32_t val32 = (uint32_t)abs(ival32);
	uint64_t tmp64[4];
	uint32_t* tmp = (uint32_t*)tmp64;
	uint64_t* r32 = (uint64_t*)res;
	RCF_mul_wide_32(r32[0], val288[0], val32);
	RCF_mul_wide_32(r32[1], val288[2], val32);
	RCF_mul_wide_32(r32[2], val288[4], val32);
	RCF_mul_wide_32(r32[3], val288[6], val32);
	RCF_mul_wide_32(tmp64[0], val288[1], val32);
	RCF_mul_wide_32(tmp64[1], val288[3], val32);
	RCF_mul_wide_32(tmp64[2], val288[5], val32);
	RCF_mul_wide_32(tmp64[3], val288[7], val32);

	RCF_add_cc_32(res[1], res[1], tmp[0]);
	RCF_addc_cc_32(res[2], res[2], tmp[1]);
	RCF_addc_cc_32(res[3], res[3], tmp[2]);
	RCF_addc_cc_32(res[4], res[4], tmp[3]);
	RCF_addc_cc_32(res[5], res[5], tmp[4]);
	RCF_addc_cc_32(res[6], res[6], tmp[5]);
	RCF_addc_cc_32(res[7], res[7], tmp[6]);
	RCF_madc_lo_32(res[8], val288[8], val32, tmp[7]);

	if (ival32 < 0)
		inv_neg_288(res);
}

__device__ __forceinline__ void inv_set_288_i32(uint32_t* res, int val)
{
	res[0] = (uint32_t)val;
	res[1] = (val < 0) ? 0xFFFFFFFFu : 0u;
	res[2] = res[1];
	res[3] = res[1];
	res[4] = res[1];
	res[5] = res[1];
	res[6] = res[1];
	res[7] = res[1];
	res[8] = res[1];
}

// res = P * val, as a 288-bit value. P = 2^256 - (2^32 + 977), so this is
// val*2^256 - val*(2^32 + 977).
__device__ __forceinline__ void inv_mul_P_by_32(uint32_t* res, uint32_t val)
{
	__align__(8) uint32_t tmp[3];
	RCF_mul_wide_32(*(uint64_t*)tmp, val, RCF_P_INV32);
	RCF_add_cc_32(tmp[1], tmp[1], val);
	RCF_addc_32(tmp[2], 0, 0);

	RCF_sub_cc_32(res[0], 0, tmp[0]);
	RCF_subc_cc_32(res[1], 0, tmp[1]);
	RCF_subc_cc_32(res[2], 0, tmp[2]);
	RCF_subc_cc_32(res[3], 0, 0);
	RCF_subc_cc_32(res[4], 0, 0);
	RCF_subc_cc_32(res[5], 0, 0);
	RCF_subc_cc_32(res[6], 0, 0);
	RCF_subc_cc_32(res[7], 0, 0);
	RCF_subc_32(res[8], val, 0);
}

__device__ __forceinline__ void inv_shiftR_288_by_30(uint32_t* res)
{
	res[0] = __funnelshift_r(res[0], res[1], 30);
	res[1] = __funnelshift_r(res[1], res[2], 30);
	res[2] = __funnelshift_r(res[2], res[3], 30);
	res[3] = __funnelshift_r(res[3], res[4], 30);
	res[4] = __funnelshift_r(res[4], res[5], 30);
	res[5] = __funnelshift_r(res[5], res[6], 30);
	res[6] = __funnelshift_r(res[6], res[7], 30);
	res[7] = __funnelshift_r(res[7], res[8], 30);
	res[8] = (uint32_t)(((int)res[8]) >> 30);
}

__device__ __forceinline__ void inv_add_288_P(uint32_t* res)
{
	RCF_add_cc_32(res[0], res[0], 0xFFFFFC2Fu);
	RCF_addc_cc_32(res[1], res[1], 0xFFFFFFFEu);
	RCF_addc_cc_32(res[2], res[2], 0xFFFFFFFFu);
	RCF_addc_cc_32(res[3], res[3], 0xFFFFFFFFu);
	RCF_addc_cc_32(res[4], res[4], 0xFFFFFFFFu);
	RCF_addc_cc_32(res[5], res[5], 0xFFFFFFFFu);
	RCF_addc_cc_32(res[6], res[6], 0xFFFFFFFFu);
	RCF_addc_cc_32(res[7], res[7], 0xFFFFFFFFu);
	RCF_addc_32(res[8], res[8], 0u);
}

__device__ __forceinline__ void inv_sub_288_P(uint32_t* res)
{
	RCF_sub_cc_32(res[0], res[0], 0xFFFFFC2Fu);
	RCF_subc_cc_32(res[1], res[1], 0xFFFFFFFEu);
	RCF_subc_cc_32(res[2], res[2], 0xFFFFFFFFu);
	RCF_subc_cc_32(res[3], res[3], 0xFFFFFFFFu);
	RCF_subc_cc_32(res[4], res[4], 0xFFFFFFFFu);
	RCF_subc_cc_32(res[5], res[5], 0xFFFFFFFFu);
	RCF_subc_cc_32(res[6], res[6], 0xFFFFFFFFu);
	RCF_subc_cc_32(res[7], res[7], 0xFFFFFFFFu);
	RCF_subc_32(res[8], res[8], 0u);
}

#define RCF_APPLY_DIV_SHIFT() \
	matrix[0] <<= index; matrix[1] <<= index; kbnt -= index; _val >>= index;

#define RCF_DO_INV_STEP() { \
	kbnt = -kbnt; int tmp_s = -_modp; _modp = _val; _val = tmp_s; \
	tmp_s = -matrix[0]; matrix[0] = matrix[2]; matrix[2] = tmp_s; \
	tmp_s = -matrix[1]; matrix[1] = matrix[3]; matrix[3] = tmp_s; }

// res = res^-1 mod p, in place. res must hold >= 9 u32 words.
__device__ __forceinline__ void InvModP(uint32_t* res)
{
	int matrix[4], _val, _modp, index, cnt, mx, kbnt;
	__align__(8) uint32_t modp[9];
	__align__(8) uint32_t val[9];
	__align__(8) uint32_t a[9];
	__align__(8) uint32_t tmp[4][9 + 1]; // +1 keeps tmp[>0] 64-bit aligned

	((uint64_t*)modp)[0] = RCF_P_0;
	((uint64_t*)modp)[1] = RCF_P_123;
	((uint64_t*)modp)[2] = RCF_P_123;
	((uint64_t*)modp)[3] = RCF_P_123;
	modp[8] = 0;
	res[8] = 0;
	val[0] = res[0]; val[1] = res[1]; val[2] = res[2]; val[3] = res[3];
	val[4] = res[4]; val[5] = res[5]; val[6] = res[6]; val[7] = res[7];
	val[8] = 0;
	matrix[0] = matrix[3] = 1;
	matrix[1] = matrix[2] = 0;
	kbnt = -1;
	_val = (int)res[0];
	_modp = (int)(uint32_t)RCF_P_0;
	index = __ffs(_val | 0x40000000) - 1;
	RCF_APPLY_DIV_SHIFT();
	cnt = 30 - index;
	while (cnt > 0)
	{
		if (kbnt < 0)
			RCF_DO_INV_STEP();
		mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt;
		int mul = (-_modp * _val) & 7;
		mul &= (int)(0xFFFFFFFFu >> mx);
		_val += _modp * mul;
		matrix[2] += matrix[0] * mul;
		matrix[3] += matrix[1] * mul;
		index = __ffs(_val | (1 << cnt)) - 1;
		RCF_APPLY_DIV_SHIFT();
		cnt -= index;
	}
	inv_mul_288_by_i32(tmp[0], modp, matrix[0]);
	inv_mul_288_by_i32(tmp[1], val, matrix[1]);
	inv_mul_288_by_i32(tmp[2], modp, matrix[2]);
	inv_mul_288_by_i32(tmp[3], val, matrix[3]);
	inv_add_288(modp, tmp[0], tmp[1]);
	inv_shiftR_288_by_30(modp);
	inv_add_288(val, tmp[2], tmp[3]);
	inv_shiftR_288_by_30(val);
	inv_set_288_i32(tmp[1], matrix[1]);
	inv_set_288_i32(tmp[3], matrix[3]);
	inv_mul_P_by_32(res, (tmp[1][0] * 0xD2253531u) & 0x3FFFFFFFu);
	inv_add_288(res, res, tmp[1]);
	inv_shiftR_288_by_30(res);
	inv_mul_P_by_32(a, (tmp[3][0] * 0xD2253531u) & 0x3FFFFFFFu);
	inv_add_288(a, a, tmp[3]);
	inv_shiftR_288_by_30(a);
	while (1)
	{
		matrix[0] = matrix[3] = 1;
		matrix[1] = matrix[2] = 0;
		_val = (int)val[0];
		_modp = (int)modp[0];
		index = __ffs(_val | 0x40000000) - 1;
		RCF_APPLY_DIV_SHIFT();
		cnt = 30 - index;
		while (cnt > 0)
		{
			if (kbnt < 0)
				RCF_DO_INV_STEP();
			mx = (kbnt + 1 < cnt) ? 31 - kbnt : 32 - cnt;
			int mul = (-_modp * _val) & 7;
			mul &= (int)(0xFFFFFFFFu >> mx);
			_val += _modp * mul;
			matrix[2] += matrix[0] * mul;
			matrix[3] += matrix[1] * mul;
			index = __ffs(_val | (1 << cnt)) - 1;
			RCF_APPLY_DIV_SHIFT();
			cnt -= index;
		}
		inv_mul_288_by_i32(tmp[0], modp, matrix[0]);
		inv_mul_288_by_i32(tmp[1], val, matrix[1]);
		inv_mul_288_by_i32(tmp[2], modp, matrix[2]);
		inv_mul_288_by_i32(tmp[3], val, matrix[3]);
		inv_add_288(modp, tmp[0], tmp[1]);
		inv_shiftR_288_by_30(modp);
		inv_add_288(val, tmp[2], tmp[3]);
		inv_shiftR_288_by_30(val);
		inv_mul_288_by_i32(tmp[0], res, matrix[0]);
		inv_mul_288_by_i32(tmp[1], a, matrix[1]);

		if ((val[0] | val[1] | val[2] | val[3] | val[4] | val[5] | val[6] | val[7]) == 0)
			break;

		inv_mul_288_by_i32(tmp[2], res, matrix[2]);
		inv_mul_288_by_i32(tmp[3], a, matrix[3]);
		inv_mul_P_by_32(res, ((tmp[0][0] + tmp[1][0]) * 0xD2253531u) & 0x3FFFFFFFu);
		inv_add_288(res, res, tmp[0]);
		inv_add_288(res, res, tmp[1]);
		inv_shiftR_288_by_30(res);
		inv_mul_P_by_32(a, ((tmp[2][0] + tmp[3][0]) * 0xD2253531u) & 0x3FFFFFFFu);
		inv_add_288(a, a, tmp[2]);
		inv_add_288(a, a, tmp[3]);
		inv_shiftR_288_by_30(a);
	}
	inv_mul_P_by_32(res, ((tmp[0][0] + tmp[1][0]) * 0xD2253531u) & 0x3FFFFFFFu);
	inv_add_288(res, res, tmp[0]);
	inv_add_288(res, res, tmp[1]);
	inv_shiftR_288_by_30(res);
	if ((int)modp[8] < 0)
		inv_neg_288(res);
	while ((int)res[8] < 0)
		inv_add_288_P(res);
	while ((int)res[8] > 0)
		inv_sub_288_P(res);
}

#endif // CUDACYCLONE_RC_INV

} // namespace rcfield
