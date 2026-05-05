
#include <cuda_runtime.h>
#if defined(_WIN32)
#include <device_launch_parameters.h>
#endif
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <thread>
#include <chrono>
#include <cmath>
#include <csignal>
#include <atomic>
#include <random>
#include <algorithm>

#include "CUDAMath.h"
#include "sha256.h"
#include "CUDAHash.cuh"
#include "CUDAUtils.h"
#include "CUDAStructures.h"

static inline bool lt256(const uint64_t a[4], const uint64_t b[4]) {
    for (int i = 3; i >= 0; --i) {
        if (a[i] < b[i]) return true;
        if (a[i] > b[i]) return false;
    }
    return false;
}

static inline void sub256_u64_host_inplace(uint64_t a[4], uint64_t dec) {
    uint64_t borrow = dec;
    for (int i = 0; i < 4 && borrow; ++i) {
        uint64_t old = a[i];
        a[i] = old - borrow;
        borrow = (old < borrow) ? 1ull : 0ull;
    }
}

static inline bool is_zero_256_host(const uint64_t a[4]) {
    return (a[0] | a[1] | a[2] | a[3]) == 0ull;
}

static inline bool eq256_host(const uint64_t a[4], const uint64_t b[4]) {
    return a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3];
}

static inline void wrap_mod_range(uint64_t value[4], const uint64_t range_len[4]) {
    if (is_zero_256_host(range_len)) return;
    while (!lt256(value, range_len)) {
        uint64_t next[4];
        sub256(value, range_len, next);
        value[0]=next[0]; value[1]=next[1]; value[2]=next[2]; value[3]=next[3];
    }
}

static inline void random_segment_start(uint64_t out[4],
                                        const uint64_t sweep_origin[4],
                                        const uint64_t global_offset[4],
                                        const uint64_t range_start[4],
                                        const uint64_t range_len[4]) {
    uint64_t rel[4];
    sub256(sweep_origin, range_start, rel);
    add256(rel, global_offset, rel);
    wrap_mod_range(rel, range_len);
    add256(range_start, rel, out);
}

static inline void advance_sweep_origin(uint64_t origin[4],
                                        const uint64_t sweep_coverage[4],
                                        const uint64_t range_start[4],
                                        const uint64_t range_len[4]) {
    uint64_t next[4];
    random_segment_start(next, origin, sweep_coverage, range_start, range_len);
    origin[0]=next[0]; origin[1]=next[1]; origin[2]=next[2]; origin[3]=next[3];
}

static inline void gen_random_256(uint64_t out[4], uint64_t lo[4], uint64_t hi[4]) {
    static thread_local std::mt19937_64 gen([]{
        std::random_device rd;
        std::seed_seq seq{
            rd(), rd(), rd(), rd(),
            (uint32_t)std::chrono::high_resolution_clock::now().time_since_epoch().count(),
            (uint32_t)((uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count() >> 32)
        };
        return std::mt19937_64(seq);
    }());
    std::uniform_int_distribution<uint64_t> dist(0, UINT64_MAX);

    uint64_t len[4];
    sub256(hi, lo, len);
    add256_u64(len, 1ull, len);

    if ((len[0] | len[1] | len[2] | len[3]) == 0ull) {
        uint64_t offset[4] = { dist(gen), dist(gen), dist(gen), dist(gen) };
        add256(lo, offset, out);
        return;
    }

    int top_limb = 3;
    while (top_limb > 0 && len[top_limb] == 0ull) --top_limb;

    int highest_bit = 63;
    while (highest_bit > 0 && ((len[top_limb] >> highest_bit) & 1ull) == 0ull) --highest_bit;
    uint64_t top_mask = highest_bit == 63 ? UINT64_MAX : ((1ull << (highest_bit + 1)) - 1ull);

    uint64_t offset[4];
    do {
        for (int i = 0; i < 4; ++i) offset[i] = 0ull;
        for (int i = 0; i <= top_limb; ++i) offset[i] = dist(gen);
        offset[top_limb] &= top_mask;
    } while (!lt256(offset, len));

    add256(lo, offset, out);
}

static volatile sig_atomic_t g_sigint = 0;
static void handle_sigint(int sig) { 
    g_sigint = 1; 
}

static inline bool cuda_check(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(e) << "\n";
        std::exit(EXIT_FAILURE);
    }
    return true;
}
#define ck(e, msg) cuda_check(e, msg)

struct GPUContext {
    int deviceId;
    cudaDeviceProp prop;
    uint64_t* d_start_scalars;
    uint64_t* d_Px;
    uint64_t* d_Py;
    uint64_t* d_Rx;
    uint64_t* d_Ry;
    uint64_t* d_counts256;
    int* d_found_flag;
    FoundResult* d_found_result;
    PartialResult* d_partial_results;
    uint32_t* d_partial_count;
    uint32_t* d_partial_overflow;
    unsigned long long* d_hashes_accum;
    unsigned int* d_any_left;
    cudaEvent_t kernelDone;
    cudaStream_t stream;
    uint64_t threadsTotal;
    int blocks;
    int threadsPerBlock;
    uint64_t per_thread_cnt[4];
    uint64_t range_start[4];
    uint64_t range_len[4];
    uint64_t launchesCompleted;
    uint64_t* d_current_scalar;
};

static constexpr uint32_t PARTIAL_RESULT_CAPACITY = 65536u;

static inline std::string formatHash160Hex(const uint8_t hash160[20]) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (int i = 0; i < 20; ++i) {
        oss << std::setw(2) << (unsigned int)hash160[i];
    }
    return oss.str();
}

static inline std::string formatCompressedPubHexFromPrefixX(uint8_t prefix, const uint64_t X[4]) {
    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::nouppercase;
    oss << std::setw(2) << (unsigned int)prefix;
    for (int limb = 3; limb >= 0; --limb) {
        oss << std::setw(16) << X[limb];
    }
    return oss.str();
}

static inline std::string partialOutputFile(uint32_t requested_chars, uint32_t match_chars) {
    if (match_chars <= requested_chars) return "partial.txt";
    return "partialp" + std::to_string(match_chars - requested_chars) + ".txt";
}

__device__ __forceinline__ int load_found_flag_relaxed(const int* p) {
    return *((const volatile int*)p);
}
__device__ __forceinline__ bool warp_found_ready(const int* __restrict__ d_found_flag,
                                                 unsigned full_mask,
                                                 unsigned lane)
{
    int f = 0;
    if (lane == 0) f = load_found_flag_relaxed(d_found_flag);
    f = __shfl_sync(full_mask, f, 0);
    return f == FOUND_READY;
}

__device__ __forceinline__ uint8_t high_nibble(uint8_t v) { return (uint8_t)(v >> 4); }
__device__ __forceinline__ uint8_t low_nibble(uint8_t v)  { return (uint8_t)(v & 0x0Fu); }

__device__ __forceinline__ uint32_t hash160_matching_hex_chars(const uint8_t* __restrict__ h) {
    uint32_t chars = 0;
#pragma unroll
    for (int i = 0; i < 20; ++i) {
        if (high_nibble(h[i]) != high_nibble(c_target_hash160[i])) return chars;
        ++chars;
        if (low_nibble(h[i]) != low_nibble(c_target_hash160[i])) return chars;
        ++chars;
    }
    return chars;
}

__device__ __forceinline__ void record_partial_result(
    uint32_t partial_chars,
    const uint8_t* __restrict__ h20,
    const uint64_t scalar[4],
    const uint64_t X[4],
    uint8_t pubkey_prefix,
    uint64_t gid,
    PartialResult* __restrict__ partial_results,
    uint32_t* __restrict__ partial_count,
    uint32_t* __restrict__ partial_overflow,
    uint32_t partial_capacity)
{
    if (partial_chars == 0 || partial_results == nullptr || partial_count == nullptr) return;
    uint32_t matched = hash160_matching_hex_chars(h20);
    if (matched < partial_chars) return;

    uint32_t pos = atomicAdd(partial_count, 1u);
    if (pos >= partial_capacity) {
        if (partial_overflow != nullptr) atomicAdd(partial_overflow, 1u);
        return;
    }

    PartialResult* out = &partial_results[pos];
    out->match_chars = matched;
    out->threadId = (uint32_t)gid;
    out->pubkey_prefix = pubkey_prefix;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        out->scalar[k] = scalar[k];
        out->X[k] = X[k];
    }
#pragma unroll
    for (int k = 0; k < 20; ++k) out->hash160[k] = h20[k];
}

#ifndef MAX_BATCH_SIZE
#define MAX_BATCH_SIZE 1024
#endif
#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

__constant__ uint64_t c_Gx[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Gy[(MAX_BATCH_SIZE/2) * 4];
__constant__ uint64_t c_Jx[4];
__constant__ uint64_t c_Jy[4];

__launch_bounds__(256, 2)
__global__ void kernel_point_add_and_check_oneinv(
    const uint64_t* __restrict__ Px,
    const uint64_t* __restrict__ Py,
    uint64_t* __restrict__ Rx,
    uint64_t* __restrict__ Ry,
    uint64_t* __restrict__ start_scalars,
    uint64_t* __restrict__ counts256,
    uint64_t threadsTotal,
    uint32_t batch_size,
    uint32_t max_batches_per_launch,
    int* __restrict__ d_found_flag,
    FoundResult* __restrict__ d_found_result,
    PartialResult* __restrict__ d_partial_results,
    uint32_t* __restrict__ d_partial_count,
    uint32_t* __restrict__ d_partial_overflow,
    uint32_t partial_chars,
    uint32_t partial_capacity,
    unsigned long long* __restrict__ hashes_accum,
    unsigned int* __restrict__ d_any_left
)
{
    const int B = (int)batch_size;
    if (B <= 0 || (B & 1) || B > MAX_BATCH_SIZE) return;
    const int half = B >> 1;

    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= threadsTotal) return;

    const unsigned lane      = (unsigned)(threadIdx.x & (WARP_SIZE - 1));
    const unsigned full_mask = 0xFFFFFFFFu;
    if (warp_found_ready(d_found_flag, full_mask, lane)) return;

    const uint32_t target_prefix = c_target_prefix;

    unsigned int local_hashes = 0;
    #define FLUSH_THRESHOLD 65536u
    #define WARP_FLUSH_HASHES() do { \
        unsigned long long v = warp_reduce_add_ull((unsigned long long)local_hashes); \
        if (lane == 0 && v) atomicAdd(hashes_accum, v); \
        local_hashes = 0; \
    } while (0)
    #define MAYBE_WARP_FLUSH() do { if ((local_hashes & (FLUSH_THRESHOLD - 1u)) == 0u) WARP_FLUSH_HASHES(); } while (0)

    uint64_t x1[4], y1[4], S[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint64_t idx = gid * 4 + i;
        x1[i] = Px[idx];
        y1[i] = Py[idx];
        S[i]  = start_scalars[idx];   
    }
    uint64_t rem[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) rem[i] = counts256[gid*4 + i];

    if ((rem[0]|rem[1]|rem[2]|rem[3]) == 0ull) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { Rx[gid*4+i] = x1[i]; Ry[gid*4+i] = y1[i]; }
        WARP_FLUSH_HASHES(); return;
    }

    uint32_t batches_done = 0;

    while (batches_done < max_batches_per_launch && ge256_u64(rem, (uint64_t)B)) {
        if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

        {
            uint8_t h20[20];
            uint8_t prefix = (uint8_t)(y1[0] & 1ULL) ? 0x03 : 0x02;
            getHash160_33_from_limbs(prefix, x1, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();
            record_partial_result(partial_chars, h20, S, x1, prefix, gid,
                                  d_partial_results, d_partial_count,
                                  d_partial_overflow, partial_capacity);

            bool full = hash160_prefix_equals(h20, target_prefix) &&
                        hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix);
            if (__any_sync(full_mask, full)) {
                if (full) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=S[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=x1[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y1[k];
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }
        }

        uint64_t subp[MAX_BATCH_SIZE/2][4];
        uint64_t acc[4], tmp[4];

#pragma unroll
        for (int j=0;j<4;++j) acc[j] = c_Jx[j];
        ModSub256(acc, acc, x1);
#pragma unroll
        for (int j=0;j<4;++j) subp[half-1][j] = acc[j];

        for (int i = half - 2; i >= 0; --i) {
#pragma unroll
            for (int j=0;j<4;++j) tmp[j] = c_Gx[(size_t)(i+1)*4 + j];
            ModSub256(tmp, tmp, x1);
            _ModMult(acc, acc, tmp);
#pragma unroll
            for (int j=0;j<4;++j) subp[i][j] = acc[j];
        }

        uint64_t d0[4], inverse[5];
#pragma unroll
        for (int j=0;j<4;++j) d0[j] = c_Gx[0*4 + j];
        ModSub256(d0, d0, x1);
#pragma unroll
        for (int j=0;j<4;++j) inverse[j] = d0[j];
        _ModMult(inverse, subp[0]);
        inverse[4] = 0ull;
        _ModInv(inverse);

        uint64_t sy_neg[4], sx_neg[4];
        ModNeg256(sy_neg, y1);
        ModNeg256(sx_neg, x1);

        for (int i = 0; i < half - 1; ++i) {
            if (warp_found_ready(d_found_flag, full_mask, lane)) { WARP_FLUSH_HASHES(); return; }

            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);     
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3); 
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();
                uint64_t fs_partial[4]; for (int k=0;k<4;++k) fs_partial[k]=S[k];
                uint64_t addv_partial=(uint64_t)(i+1);
                for (int k=0;k<4 && addv_partial;++k){ uint64_t old=fs_partial[k]; fs_partial[k]=old+addv_partial; addv_partial=(fs_partial[k]<old)?1ull:0ull; }
                record_partial_result(partial_chars, h20, fs_partial, px3, odd?0x03:0x02, gid,
                                      d_partial_results, d_partial_count,
                                      d_partial_overflow, partial_capacity);

                bool full = hash160_prefix_equals(h20, target_prefix) &&
                            hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix);
                if (__any_sync(full_mask, full)) {
                    if (full) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t addv=(uint64_t)(i+1);
                            for (int k=0;k<4 && addv;++k){ uint64_t old=fs[k]; fs[k]=old+addv; addv=(fs[k]<old)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                           
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            {
                uint64_t px3[4], s[4], lam[4];
                uint64_t px_i[4], py_i[4];
#pragma unroll
                for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
                ModNeg256(py_i, py_i); 

                ModSub256(s, py_i, y1);
                _ModMult(lam, s, dx_inv_i);

                _ModSqr(px3, lam);
                ModSub256(px3, px3, x1);
                ModSub256(px3, px3, px_i);

                ModSub256(s, x1, px3);
                _ModMult(s, s, lam);
                uint8_t odd; ModSub256isOdd(s, y1, &odd);

                uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
                ++local_hashes; MAYBE_WARP_FLUSH();
                uint64_t fs_partial[4]; for (int k=0;k<4;++k) fs_partial[k]=S[k];
                uint64_t sub_partial=(uint64_t)(i+1);
                for (int k=0;k<4 && sub_partial;++k){ uint64_t old=fs_partial[k]; fs_partial[k]=old-sub_partial; sub_partial=(old<sub_partial)?1ull:0ull; }
                record_partial_result(partial_chars, h20, fs_partial, px3, odd?0x03:0x02, gid,
                                      d_partial_results, d_partial_count,
                                      d_partial_overflow, partial_capacity);

                bool full = hash160_prefix_equals(h20, target_prefix) &&
                            hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix);
                if (__any_sync(full_mask, full)) {
                    if (full) {
                        if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                            uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                            uint64_t sub=(uint64_t)(i+1);
                            for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                            uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                            for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                            d_found_result->threadId = (int)gid;
                            d_found_result->iter     = 0;
                            __threadfence_system();
                            atomicExch(d_found_flag, FOUND_READY);
                        }
                    }
                    __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
                }
            }

            uint64_t gxmi[4];
#pragma unroll
            for (int j=0;j<4;++j) gxmi[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(gxmi, gxmi, x1);
            _ModMult(inverse, inverse, gxmi);
        }

        {
            const int i = half - 1;
            uint64_t dx_inv_i[4];
            _ModMult(dx_inv_i, subp[i], inverse);

            uint64_t px3[4], s[4], lam[4];
            uint64_t px_i[4], py_i[4];
#pragma unroll
            for (int j=0;j<4;++j) { px_i[j]=c_Gx[(size_t)i*4+j]; py_i[j]=c_Gy[(size_t)i*4+j]; }
            ModNeg256(py_i, py_i);

            ModSub256(s, py_i, y1);
            _ModMult(lam, s, dx_inv_i);

            _ModSqr(px3, lam);
            ModSub256(px3, px3, x1);
            ModSub256(px3, px3, px_i);

            ModSub256(s, x1, px3);
            _ModMult(s, s, lam);
            uint8_t odd; ModSub256isOdd(s, y1, &odd);

            uint8_t h20[20]; getHash160_33_from_limbs(odd?0x03:0x02, px3, h20);
            ++local_hashes; MAYBE_WARP_FLUSH();
            uint64_t fs_partial[4]; for (int k=0;k<4;++k) fs_partial[k]=S[k];
            uint64_t sub_partial=(uint64_t)half;
            for (int k=0;k<4 && sub_partial;++k){ uint64_t old=fs_partial[k]; fs_partial[k]=old-sub_partial; sub_partial=(old<sub_partial)?1ull:0ull; }
            record_partial_result(partial_chars, h20, fs_partial, px3, odd?0x03:0x02, gid,
                                  d_partial_results, d_partial_count,
                                  d_partial_overflow, partial_capacity);

            bool full = hash160_prefix_equals(h20, target_prefix) &&
                        hash160_matches_prefix_then_full(h20, c_target_hash160, target_prefix);
            if (__any_sync(full_mask, full)) {
                if (full) {
                    if (atomicCAS(d_found_flag, FOUND_NONE, FOUND_LOCK) == FOUND_NONE) {
                        uint64_t fs[4]; for (int k=0;k<4;++k) fs[k]=S[k];
                        uint64_t sub=(uint64_t)half;
                        for (int k=0;k<4 && sub;++k){ uint64_t old=fs[k]; fs[k]=old-sub; sub=(old<sub)?1ull:0ull; }
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->scalar[k]=fs[k];
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Rx[k]=px3[k];
                        uint64_t y3[4]; uint64_t t[4]; ModSub256(t, x1, px3); _ModMult(y3, t, lam); ModSub256(y3, y3, y1);
#pragma unroll
                        for (int k=0;k<4;++k) d_found_result->Ry[k]=y3[k];
                        d_found_result->threadId = (int)gid;
                        d_found_result->iter     = 0;
                        __threadfence_system();
                        atomicExch(d_found_flag, FOUND_READY);
                    }
                }
                __syncwarp(full_mask); WARP_FLUSH_HASHES(); return;
            }

            uint64_t last_dx[4];
#pragma unroll
            for (int j=0;j<4;++j) last_dx[j] = c_Gx[(size_t)i*4 + j];
            ModSub256(last_dx, last_dx, x1);
            _ModMult(inverse, inverse, last_dx);
        }

        {
            uint64_t lam[4], s[4], x3[4], y3[4];

            uint64_t Jy_minus_y1[4];
#pragma unroll
            for (int j=0;j<4;++j) Jy_minus_y1[j] = c_Jy[j];
            ModSub256(Jy_minus_y1, Jy_minus_y1, y1);

            _ModMult(lam, Jy_minus_y1, inverse);
            _ModSqr(x3, lam);
            ModSub256(x3, x3, x1);
            uint64_t Jx_local[4]; for (int j=0;j<4;++j) Jx_local[j]=c_Jx[j];
            ModSub256(x3, x3, Jx_local);

            ModSub256(s, x1, x3);
            _ModMult(y3, s, lam);
            ModSub256(y3, y3, y1);

#pragma unroll
            for (int j=0;j<4;++j) { x1[j] = x3[j]; y1[j] = y3[j]; }
        }

        {
            uint64_t addv=(uint64_t)B;
            for (int k=0;k<4 && addv;++k){ uint64_t old=S[k]; S[k]=old+addv; addv=(S[k]<old)?1ull:0ull; }
            sub256_u64_inplace(rem, (uint64_t)B);
        }
        ++batches_done;
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        Rx[gid*4+i] = x1[i];
        Ry[gid*4+i] = y1[i];
        counts256[gid*4+i] = rem[i];
        start_scalars[gid*4+i] = S[i];
    }
    if ((rem[0] | rem[1] | rem[2] | rem[3]) != 0ull) {
        atomicAdd(d_any_left, 1u);
    }

    WARP_FLUSH_HASHES();
    #undef MAYBE_WARP_FLUSH
    #undef WARP_FLUSH_HASHES
    #undef FLUSH_THRESHOLD
}

extern bool hexToLE64(const std::string& h_in, uint64_t w[4]);
extern bool hexToHash160(const std::string& h, uint8_t hash160[20]);
extern std::string formatHex256(const uint64_t limbs[4]);
extern long double ld_from_u256(const uint64_t v[4]);
extern bool decode_p2pkh_address(const std::string& addr, uint8_t out20[20]);
extern std::string formatCompressedPubHex(const uint64_t X[4], const uint64_t Y[4]);
extern void add256_u64_mul(const uint64_t a[4], uint64_t mult, uint64_t out[4]);
extern void divmod_256_by_u64_array(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t remainder[4]);
__global__ void scalarMulKernelBase(const uint64_t* scalars_in, uint64_t* outX, uint64_t* outY, int N);

static inline bool parse_uint32_arg(const char* text, uint32_t min_value, uint32_t max_value, uint32_t& out) {
    if (text == nullptr || *text == '\0') return false;
    uint64_t value = 0;
    for (const char* p = text; *p != '\0'; ++p) {
        if (*p < '0' || *p > '9') return false;
        value = value * 10u + (uint32_t)(*p - '0');
        if (value > max_value) return false;
    }
    if (value < min_value) return false;
    out = (uint32_t)value;
    return true;
}

int main(int argc, char** argv) {
    std::signal(SIGINT, handle_sigint);

    std::cout <<
R"(  __  __ _   _ _  _____ ___ ______   ______ _     ___  _   _ _____
 |  \/  | | | | ||_   _|_ _/ ___\ \ / / ___| |   / _ \| \ | | ____|
 | |\/| | | | | |  | |  | | |    \ V / |   | |  | | | |  \| |  _|
 | |  | | |_| | |__| |  | | |___  | || |___| |__| |_| | |\  | |___
 |_|  |_|\___/|____|_| |___\____| |_| \____|_____\___/|_| \_|_____|

 MULTICYCLONE by Draikoon - forked from Dookoo2

)";

    std::string target_hash_hex, range_hex, address_b58;
    uint32_t runtime_points_batch_size = 128;
    uint32_t runtime_batches_per_sm    = 8;
    uint32_t slices_per_launch         = 64;
    std::string gpu_list_str;
    uint32_t random_interval_seconds   = 0;
    uint32_t partial_digits            = 0;
    bool random_mode = false;

    auto parse_grid = [](const std::string& s, uint32_t& a_out, uint32_t& b_out)->bool {
        size_t comma = s.find(',');
        if (comma == std::string::npos) return false;
        auto trim = [](std::string& z){
            size_t p1 = z.find_first_not_of(" \t");
            size_t p2 = z.find_last_not_of(" \t");
            if (p1 == std::string::npos) { z.clear(); return; }
            z = z.substr(p1, p2 - p1 + 1);
        };
        std::string a_str = s.substr(0, comma);
        std::string b_str = s.substr(comma + 1);
        trim(a_str); trim(b_str);
        if (a_str.empty() || b_str.empty()) return false;
        uint32_t aa = 0, bb = 0;
        if (!parse_uint32_arg(a_str.c_str(), 1u, (1u << 20), aa)) return false;
        if (!parse_uint32_arg(b_str.c_str(), 1u, (1u << 20), bb)) return false;
        a_out=aa; b_out=bb; return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if      ((arg == "--target-hash160" || arg == "-target-hash160") && i + 1 < argc) target_hash_hex = argv[++i];
        else if ((arg == "--address" || arg == "-address")               && i + 1 < argc) address_b58     = argv[++i];
        else if ((arg == "--range" || arg == "-range")                   && i + 1 < argc) range_hex       = argv[++i];
        else if ((arg == "--grid" || arg == "-grid")                     && i + 1 < argc) {
            uint32_t a=0,b=0;
            if (!parse_grid(argv[++i], a, b)) {
                std::cerr << "Error: --grid expects \"A,B\" (positive integers).\n";
                return EXIT_FAILURE;
            }
            runtime_points_batch_size = a;
            runtime_batches_per_sm    = b;
        }
        else if ((arg == "--slices" || arg == "-slices") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, (1u << 20), v)) {
                std::cerr << "Error: --slices must be in 1.." << (1u<<20) << "\n";
                return EXIT_FAILURE;
            }
            slices_per_launch = v;
        }
        else if ((arg == "--gpus" || arg == "-gpus") && i + 1 < argc) {
            gpu_list_str = argv[++i];
        }
        else if ((arg == "--random-interval" || arg == "-random-interval") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, 86400u, v)) {
                std::cerr << "Error: --random-interval must be 1..86400 seconds.\n";
                return EXIT_FAILURE;
            }
            random_interval_seconds = v;
            random_mode = true;
        }
        else if ((arg == "--partial" || arg == "-partial") && i + 1 < argc) {
            uint32_t v = 0;
            if (!parse_uint32_arg(argv[++i], 1u, 40u, v)) {
                std::cerr << "Error: --partial must be a number of hash160 hex digits in 1..40.\n";
                return EXIT_FAILURE;
            }
            partial_digits = v;
        }
    }

    if (range_hex.empty() || (target_hash_hex.empty() && address_b58.empty())) {
        std::cerr << "Usage: " << argv[0]
                  << " --range <start_hex>:<end_hex> (--address <base58> | --target-hash160 <hash160_hex>) [--grid A,B] [--slices N] [--gpus GPU1,GPU2,...] [--random-interval SECONDS] [--partial HEX_DIGITS]\n";
        return EXIT_FAILURE;
    }
    if (!target_hash_hex.empty() && !address_b58.empty()) {
        std::cerr << "Error: provide either --address or --target-hash160, not both.\n";
        return EXIT_FAILURE;
    }

    size_t colon_pos = range_hex.find(':');
    if (colon_pos == std::string::npos) { std::cerr << "Error: range format must be start:end\n"; return EXIT_FAILURE; }
    std::string start_hex = range_hex.substr(0, colon_pos);
    std::string end_hex   = range_hex.substr(colon_pos + 1);
    
    
    while (start_hex.length() < 64) start_hex = "0" + start_hex;
    while (end_hex.length() < 64) end_hex = "0" + end_hex;
    
    if (start_hex.length() > 64) start_hex = start_hex.substr(start_hex.length() - 64);
    if (end_hex.length() > 64) end_hex = end_hex.substr(end_hex.length() - 64);

    uint64_t range_start[4]{0}, range_end[4]{0};
    if (!hexToLE64(start_hex, range_start) || !hexToLE64(end_hex, range_end)) {
        std::cerr << "Error: invalid range hex\n"; return EXIT_FAILURE;
    }

    std::cout << "Parsed range: " << formatHex256(range_start) << " - " << formatHex256(range_end) << std::endl;

    uint8_t target_hash160[20];
    if (!address_b58.empty()) {
        if (!decode_p2pkh_address(address_b58, target_hash160)) {
            std::cerr << "Error: invalid P2PKH address\n"; return EXIT_FAILURE;
        }
    } else {
        if (!hexToHash160(target_hash_hex, target_hash160)) {
            std::cerr << "Error: invalid target hash160 hex\n"; return EXIT_FAILURE;
        }
    }

    auto is_pow2 = [](uint32_t v)->bool { return v && ((v & (v-1)) == 0); };
    if (!is_pow2(runtime_points_batch_size) || (runtime_points_batch_size & 1u)) {
        std::cerr << "Error: batch size must be even and a power of two.\n";
        return EXIT_FAILURE;
    }
    if (runtime_points_batch_size > MAX_BATCH_SIZE) {
        std::cerr << "Error: batch size must be <= " << MAX_BATCH_SIZE << " (kernel limit).\n";
        return EXIT_FAILURE;
    }

    uint64_t range_len[4]; sub256(range_end, range_start, range_len); add256_u64(range_len, 1ull, range_len);

    int device=0; cudaDeviceProp prop{};
    if (cudaGetDevice(&device)!=cudaSuccess || cudaGetDeviceProperties(&prop, device)!=cudaSuccess) {
        std::cerr<<"CUDA init error\n"; return EXIT_FAILURE;
    }

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        std::cerr << "Error getting CUDA device count: " << cudaGetErrorString(err) << "\n";
        return EXIT_FAILURE;
    }
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA devices found\n";
        return EXIT_FAILURE;
    }

    std::vector<int> selectedDevices;
    if (!gpu_list_str.empty()) {
        std::stringstream ss(gpu_list_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            uint32_t parsedDev = 0;
            if (!parse_uint32_arg(item.c_str(), 0u, (uint32_t)deviceCount - 1u, parsedDev)) {
                std::cerr << "Error: Invalid device '" << item << "' (max " << deviceCount-1 << ")\n";
                return EXIT_FAILURE;
            }
            int dev = (int)parsedDev;
            if (dev < 0 || dev >= deviceCount) {
                std::cerr << "Error: Invalid device " << dev << " (max " << deviceCount-1 << ")\n";
                return EXIT_FAILURE;
            }
            selectedDevices.push_back(dev);
        }
    } else {
        for (int i = 0; i < deviceCount; ++i) selectedDevices.push_back(i);
    }

    int numGPUs = (int)selectedDevices.size();
    std::cout << "======== Multi-GPU Configuration =======================\n";
    for (int i = 0; i < numGPUs; ++i) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, selectedDevices[i]);
        std::cout << "  GPU " << i << ": " << p.name << " (compute " << p.major << "." << p.minor << ", " 
                  << p.multiProcessorCount << " SMs, " << human_bytes((double)p.totalGlobalMem) << ")\n";
    }
    std::cout << "======================================================== \n\n";

    std::vector<GPUContext> gpus;
    gpus.resize(numGPUs);

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        int devId = selectedDevices[gpuIdx];
        cudaSetDevice(devId);
        gpus[gpuIdx].deviceId = devId;
        cudaGetDeviceProperties(&gpus[gpuIdx].prop, devId);
    }

    uint64_t range_per_gpu[4];
    uint64_t range_remainder_arr[4];
    divmod_256_by_u64_array(range_len, (uint64_t)numGPUs, range_per_gpu, range_remainder_arr);
    uint64_t range_remainder = range_remainder_arr[0];

uint64_t total_threads_single = 0;
    {
        uint64_t q_div_batch[4], r_div_batch = 0ull;
        divmod_256_by_u64(range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
        bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
        uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, selectedDevices[0]);
        int threadsPerBlock = 256;
        if (threadsPerBlock > (int)prop.maxThreadsPerBlock) threadsPerBlock = prop.maxThreadsPerBlock;
        if (threadsPerBlock < 32) threadsPerBlock = 32;
        
        uint64_t userUpper = (uint64_t)prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
        if (userUpper == 0ull) userUpper = UINT64_MAX;
        
        auto pick_threads_total = [&](uint64_t upper)->uint64_t {
            if (upper < (uint64_t)threadsPerBlock) return 0ull;
            uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
            uint64_t q = total_batches_u64;
            while (t >= (uint64_t)threadsPerBlock) {
                if ((q % t) == 0ull) return t;
                t -= (uint64_t)threadsPerBlock;
            }
            return 0ull;
        };
        
        uint64_t maxThreadsByMem = (prop.totalGlobalMem > 64ull*1024*1024) ? (prop.totalGlobalMem - 64ull*1024*1024) / (2ull*4ull*sizeof(uint64_t)) : (prop.totalGlobalMem / (2ull*4ull*sizeof(uint64_t)));
        
        uint64_t upper = maxThreadsByMem;
        if (total_batches_u64 < upper) upper = total_batches_u64;
        if (userUpper < upper) upper = userUpper;
        
        total_threads_single = pick_threads_total(upper);
    }

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);

        uint64_t gpu_range_len[4];
        for (int k = 0; k < 4; ++k) gpu_range_len[k] = range_per_gpu[k];
        if ((uint64_t)gpuIdx < range_remainder) {
            add256_u64(gpu_range_len, 1ull, gpu_range_len);
        }
        for (int k = 0; k < 4; ++k) gpu.range_len[k] = gpu_range_len[k];

        uint64_t start[4];
        for (int k = 0; k < 4; ++k) start[k] = range_start[k];
        uint64_t offset[4];
        for (int k = 0; k < 4; ++k) offset[k] = range_per_gpu[k];
        add256_u64_mul(offset, (uint64_t)gpuIdx, offset);
        add256(start, offset, start);
        for (int k = 0; k < 4; ++k) gpu.range_start[k] = start[k];

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        int threadsPerBlock = 256;
        if (threadsPerBlock > (int)gpu.prop.maxThreadsPerBlock) threadsPerBlock = gpu.prop.maxThreadsPerBlock;
        if (threadsPerBlock < 32) threadsPerBlock = 32;
        gpu.threadsPerBlock = threadsPerBlock;

        const uint64_t bytesPerThread = 2ull*4ull*sizeof(uint64_t);
        size_t totalGlobalMem = gpu.prop.totalGlobalMem;
        const uint64_t reserveBytes = 64ull * 1024 * 1024;
        uint64_t usableMem = (totalGlobalMem > reserveBytes) ? (totalGlobalMem - reserveBytes) : (totalGlobalMem / 2);
        uint64_t maxThreadsByMem = usableMem / bytesPerThread;

        uint64_t q_div_batch[4], r_div_batch = 0ull;
        divmod_256_by_u64(gpu_range_len, (uint64_t)runtime_points_batch_size, q_div_batch, r_div_batch);
        bool q_fits_u64 = (q_div_batch[3]|q_div_batch[2]|q_div_batch[1]) == 0ull;
        uint64_t total_batches_u64 = q_fits_u64 ? q_div_batch[0] : 0ull;
        if (!q_fits_u64) { 
            total_batches_u64 = UINT64_MAX;
        }

        uint64_t userUpper = (uint64_t)gpu.prop.multiProcessorCount * (uint64_t)runtime_batches_per_sm * (uint64_t)threadsPerBlock;
        if (userUpper == 0ull) userUpper = UINT64_MAX;

        auto pick_threads_total = [&](uint64_t upper)->uint64_t {
            if (upper < (uint64_t)threadsPerBlock) return 0ull;
            uint64_t t = upper - (upper % (uint64_t)threadsPerBlock);
            uint64_t q = total_batches_u64;
            while (t >= (uint64_t)threadsPerBlock) {
                if (q >= t && (q % t) == 0ull) return t;
                t -= (uint64_t)threadsPerBlock;
            }
            
            t = upper - (upper % (uint64_t)threadsPerBlock);
            if (t >= (uint64_t)threadsPerBlock) return t;
            return 0ull;
        };

        uint64_t upper = maxThreadsByMem;
        if (total_batches_u64 < upper) upper = total_batches_u64;
        if (userUpper < upper) upper = userUpper;

        uint64_t threadsTotal = pick_threads_total(upper);
        if (threadsTotal == 0ull) {
            threadsTotal = (uint64_t)threadsPerBlock;
        }
        gpu.threadsTotal = threadsTotal;
        gpu.blocks = (int)(threadsTotal / (uint64_t)threadsPerBlock);

        uint64_t per_thread_cnt[4]; uint64_t r_u64 = 0ull;
        divmod_256_by_u64(gpu_range_len, threadsTotal, per_thread_cnt, r_u64);
        if (r_u64 != 0ull) {
            add256_u64(per_thread_cnt, 1ull, per_thread_cnt);
        }
        {   uint64_t qq[4], rr=0ull;
            divmod_256_by_u64(per_thread_cnt, (uint64_t)runtime_points_batch_size, qq, rr);
            if (rr != 0ull) {
                uint64_t batch_mult[4] = { (uint64_t)runtime_points_batch_size, 0, 0, 0 };
                add256(per_thread_cnt, batch_mult, per_thread_cnt);
            }
        }
        for (int k = 0; k < 4; ++k) gpu.per_thread_cnt[k] = per_thread_cnt[k];
    }

    const uint32_t B = runtime_points_batch_size;
    const uint32_t half = B >> 1;
    uint64_t random_sweep_origin[4] = {0, 0, 0, 0};
    uint64_t random_global_offset[4] = {0, 0, 0, 0};
    uint64_t random_sweep_coverage[4] = {0, 0, 0, 0};
    if (random_mode) {
        gen_random_256(random_sweep_origin, range_start, range_end);
        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            uint64_t gpu_coverage[4];
            add256_u64_mul(gpus[gpuIdx].per_thread_cnt, gpus[gpuIdx].threadsTotal, gpu_coverage);
            add256(random_sweep_coverage, gpu_coverage, random_sweep_coverage);
        }
        if (!is_zero_256_host(range_len) && lt256(range_len, random_sweep_coverage) && !eq256_host(range_len, random_sweep_coverage)) {
            std::cout << "RANDOM MODE WARNING: configured sweep coverage is larger than the requested range; overlap is unavoidable.\n";
        }
    }

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);

        uint64_t* h_counts256 = nullptr;
        uint64_t* h_start_scalars = nullptr;
        cudaHostAlloc(&h_counts256,     gpu.threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
        cudaHostAlloc(&h_start_scalars, gpu.threadsTotal * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);

        for (uint64_t i = 0; i < gpu.threadsTotal; ++i) {
            h_counts256[i*4+0] = gpu.per_thread_cnt[0];
            h_counts256[i*4+1] = gpu.per_thread_cnt[1];
            h_counts256[i*4+2] = gpu.per_thread_cnt[2];
            h_counts256[i*4+3] = gpu.per_thread_cnt[3];
        }

        {
            uint64_t cur[4] = { gpu.range_start[0], gpu.range_start[1], gpu.range_start[2], gpu.range_start[3] };
            uint64_t cur_random_offset[4] = { random_global_offset[0], random_global_offset[1], random_global_offset[2], random_global_offset[3] };
            for (uint64_t i = 0; i < gpu.threadsTotal; ++i) {
                uint64_t base[4];
                if (random_mode) {
                    random_segment_start(base, random_sweep_origin, cur_random_offset, range_start, range_len);
                } else {
                    base[0]=cur[0]; base[1]=cur[1]; base[2]=cur[2]; base[3]=cur[3];
                }
                uint64_t Sc[4]; add256_u64(base, (uint64_t)half, Sc); 
                h_start_scalars[i*4+0] = Sc[0];
                h_start_scalars[i*4+1] = Sc[1];
                h_start_scalars[i*4+2] = Sc[2];
                h_start_scalars[i*4+3] = Sc[3];

                if (!random_mode) {
                    uint64_t next[4]; add256(cur, gpu.per_thread_cnt, next);
                    cur[0]=next[0]; cur[1]=next[1]; cur[2]=next[2]; cur[3]=next[3];
                } else {
                    uint64_t next[4]; add256(cur_random_offset, gpu.per_thread_cnt, next);
                    cur_random_offset[0]=next[0]; cur_random_offset[1]=next[1]; cur_random_offset[2]=next[2]; cur_random_offset[3]=next[3];
                }
            }
            if (random_mode) {
                random_global_offset[0]=cur_random_offset[0]; random_global_offset[1]=cur_random_offset[1];
                random_global_offset[2]=cur_random_offset[2]; random_global_offset[3]=cur_random_offset[3];
            }
        }

        ck(cudaMalloc(&gpu.d_start_scalars, gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_start_scalars)");
        ck(cudaMalloc(&gpu.d_Px,           gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Px)");
        ck(cudaMalloc(&gpu.d_Py,           gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Py)");
        ck(cudaMalloc(&gpu.d_Rx,           gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Rx)");
        ck(cudaMalloc(&gpu.d_Ry,           gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_Ry)");
        ck(cudaMalloc(&gpu.d_counts256,    gpu.threadsTotal * 4 * sizeof(uint64_t)), "cudaMalloc(d_counts256)");
        ck(cudaMalloc(&gpu.d_found_flag,   sizeof(int)),                         "cudaMalloc(d_found_flag)");
        ck(cudaMalloc(&gpu.d_found_result, sizeof(FoundResult)),                 "cudaMalloc(d_found_result)");
        ck(cudaMalloc(&gpu.d_partial_results, (size_t)PARTIAL_RESULT_CAPACITY * sizeof(PartialResult)), "cudaMalloc(d_partial_results)");
        ck(cudaMalloc(&gpu.d_partial_count,    sizeof(uint32_t)),                 "cudaMalloc(d_partial_count)");
        ck(cudaMalloc(&gpu.d_partial_overflow, sizeof(uint32_t)),                 "cudaMalloc(d_partial_overflow)");
        ck(cudaMalloc(&gpu.d_hashes_accum, sizeof(unsigned long long)),          "cudaMalloc(d_hashes_accum)");
        ck(cudaMalloc(&gpu.d_any_left,     sizeof(unsigned int)),                "cudaMalloc(d_any_left)");

        ck(cudaMemcpy(gpu.d_start_scalars, h_start_scalars, gpu.threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy start_scalars");
        ck(cudaMemcpy(gpu.d_counts256,     h_counts256,     gpu.threadsTotal * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy counts256");
        { int zero = FOUND_NONE; unsigned long long zero64=0ull; unsigned int zeroU=0u;
          ck(cudaMemcpy(gpu.d_found_flag, &zero,   sizeof(int),                cudaMemcpyHostToDevice), "init found_flag");
          ck(cudaMemcpy(gpu.d_partial_count, &zeroU, sizeof(uint32_t),          cudaMemcpyHostToDevice), "init partial_count");
          ck(cudaMemcpy(gpu.d_partial_overflow, &zeroU, sizeof(uint32_t),       cudaMemcpyHostToDevice), "init partial_overflow");
          ck(cudaMemcpy(gpu.d_hashes_accum, &zero64, sizeof(unsigned long long), cudaMemcpyHostToDevice), "init hashes_accum");
          ck(cudaMemcpy(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice), "init any_left"); }

        cudaFreeHost(h_counts256);
        cudaFreeHost(h_start_scalars);

        {
            int blocks_scal = (int)((gpu.threadsTotal + gpu.threadsPerBlock - 1) / gpu.threadsPerBlock);
            scalarMulKernelBase<<<blocks_scal, gpu.threadsPerBlock>>>(gpu.d_start_scalars, gpu.d_Px, gpu.d_Py, (int)gpu.threadsTotal);
            ck(cudaDeviceSynchronize(), "scalarMulKernelBase sync");
            ck(cudaGetLastError(), "scalarMulKernelBase launch");
        }

        {
            uint64_t* h_scalars_half = nullptr;
            cudaHostAlloc(&h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
            std::memset(h_scalars_half, 0, (size_t)half * 4 * sizeof(uint64_t));
            for (uint32_t k = 0; k < half; ++k) h_scalars_half[(size_t)k*4 + 0] = (uint64_t)(k + 1);

            uint64_t *d_scalars_half=nullptr, *d_Gx_half=nullptr, *d_Gy_half=nullptr;
            ck(cudaMalloc(&d_scalars_half, (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_scalars_half)");
            ck(cudaMalloc(&d_Gx_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gx_half)");
            ck(cudaMalloc(&d_Gy_half,      (size_t)half * 4 * sizeof(uint64_t)), "cudaMalloc(d_Gy_half)");
            ck(cudaMemcpy(d_scalars_half, h_scalars_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy half scalars");

            int blocks_scal = (int)((half + gpu.threadsPerBlock - 1) / gpu.threadsPerBlock);
            scalarMulKernelBase<<<blocks_scal, gpu.threadsPerBlock>>>(d_scalars_half, d_Gx_half, d_Gy_half, (int)half);
            ck(cudaDeviceSynchronize(), "scalarMulKernelBase(half) sync");
            ck(cudaGetLastError(), "scalarMulKernelBase(half) launch");

            uint64_t* h_Gx_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
            uint64_t* h_Gy_half = (uint64_t*)std::malloc((size_t)half * 4 * sizeof(uint64_t));
            ck(cudaMemcpy(h_Gx_half, d_Gx_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gx_half");
            ck(cudaMemcpy(h_Gy_half, d_Gy_half, (size_t)half * 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Gy_half");
            ck(cudaMemcpyToSymbol(c_Gx, h_Gx_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gx");
            ck(cudaMemcpyToSymbol(c_Gy, h_Gy_half, (size_t)half * 4 * sizeof(uint64_t)), "ToSymbol c_Gy");

            cudaFree(d_scalars_half); cudaFree(d_Gx_half); cudaFree(d_Gy_half);
            cudaFreeHost(h_scalars_half);
            std::free(h_Gx_half); std::free(h_Gy_half);
        }
        {
            uint64_t* h_scalarB = nullptr;
            cudaHostAlloc(&h_scalarB, 4 * sizeof(uint64_t), cudaHostAllocWriteCombined | cudaHostAllocMapped);
            std::memset(h_scalarB, 0, 4 * sizeof(uint64_t));
            h_scalarB[0] = (uint64_t)B;

            uint64_t *d_scalarB=nullptr, *d_Jx=nullptr, *d_Jy=nullptr;
            ck(cudaMalloc(&d_scalarB, 4 * sizeof(uint64_t)), "cudaMalloc(d_scalarB)");
            ck(cudaMalloc(&d_Jx,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jx)");
            ck(cudaMalloc(&d_Jy,      4 * sizeof(uint64_t)), "cudaMalloc(d_Jy)");
            ck(cudaMemcpy(d_scalarB, h_scalarB, 4 * sizeof(uint64_t), cudaMemcpyHostToDevice), "cpy scalarB");

            scalarMulKernelBase<<<1, 1>>>(d_scalarB, d_Jx, d_Jy, 1);
            ck(cudaDeviceSynchronize(), "scalarMulKernelBase(B) sync");
            ck(cudaGetLastError(), "scalarMulKernelBase(B) launch");

            uint64_t hJx[4], hJy[4];
            ck(cudaMemcpy(hJx, d_Jx, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jx");
            ck(cudaMemcpy(hJy, d_Jy, 4 * sizeof(uint64_t), cudaMemcpyDeviceToHost), "D2H Jy");
            ck(cudaMemcpyToSymbol(c_Jx, hJx, 4 * sizeof(uint64_t)), "ToSymbol c_Jx");
            ck(cudaMemcpyToSymbol(c_Jy, hJy, 4 * sizeof(uint64_t)), "ToSymbol c_Jy");

            cudaFree(d_scalarB); cudaFree(d_Jx); cudaFree(d_Jy);
            cudaFreeHost(h_scalarB);
        }

        ck(cudaStreamCreateWithFlags(&gpu.stream, cudaStreamNonBlocking), "create stream");
        ck(cudaEventCreateWithFlags(&gpu.kernelDone, cudaEventDisableTiming), "create event");

        uint32_t prefix_le = (uint32_t)target_hash160[0]
                           | ((uint32_t)target_hash160[1] << 8)
                           | ((uint32_t)target_hash160[2] << 16)
                           | ((uint32_t)target_hash160[3] << 24);
        cudaMemcpyToSymbol(c_target_prefix, &prefix_le, sizeof(prefix_le));
        cudaMemcpyToSymbol(c_target_hash160, target_hash160, 20);
    }

    if (random_mode) {
        advance_sweep_origin(random_sweep_origin, random_sweep_coverage, range_start, range_len);
    }

    cudaSetDevice(gpus[0].deviceId);
    std::cout << "Single-GPU threads: " << total_threads_single << "\n";
    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        std::cout << "GPU " << gpuIdx << ": " << gpu.prop.name << " (compute " << gpu.prop.major << "." << gpu.prop.minor << ")\n";
        std::cout << "  SMs: " << gpu.prop.multiProcessorCount << ", Threads: " << gpu.threadsTotal 
                  << ", Blocks: " << gpu.blocks << ", ThreadsPerBlock: " << gpu.threadsPerBlock << "\n";
        uint64_t gpu_end[4];
        add256(gpu.range_start, gpu.range_len, gpu_end);
        sub256_u64_host_inplace(gpu_end, 1ull);
        std::cout << "  Range: " << formatHex256(gpu.range_start) << " - " << formatHex256(gpu_end) << "\n";
    }
    std::cout << "======================================================== \n\n";
    std::cout << "======== Phase-1: BruteForce ==========================\n";
    if (random_mode) {
        std::cout << "RANDOM MODE: Re-randomizing every " << random_interval_seconds << " seconds\n\n";
    }
    if (partial_digits != 0) {
        std::cout << "Partial hash160 saving: first " << partial_digits
                  << " hex chars to partial.txt; longer matches to partialpN.txt\n\n";
    }

    (void)cudaFuncSetCacheConfig(kernel_point_add_and_check_oneinv, cudaFuncCachePreferL1);

    auto t0 = std::chrono::high_resolution_clock::now();
    auto tLast = t0;
    unsigned long long lastHashes = 0ull;

    bool stop_all = false;
    std::atomic<bool> found_any(false);
    std::vector<unsigned long long> gpuHashes(numGPUs, 0ull);
    std::vector<bool> gpuCompleted(numGPUs, false);
    std::vector<bool> gpuNeedsLaunch(numGPUs, true);
    std::vector<uint64_t> gpuLaunches(numGPUs, 0ull);
    std::vector<uint64_t> gpuSlice(numGPUs, 0ull);
    std::vector<std::array<uint64_t,4>> gpuCurrentKey(numGPUs);
    std::vector<PartialResult> partialHost(PARTIAL_RESULT_CAPACITY);

    auto drain_partial_results = [&](GPUContext& gpu, int gpuIdx) {
        if (partial_digits == 0) return;

        uint32_t count = 0;
        uint32_t overflow = 0;
        ck(cudaMemcpy(&count, gpu.d_partial_count, sizeof(uint32_t), cudaMemcpyDeviceToHost), "read partial_count");
        ck(cudaMemcpy(&overflow, gpu.d_partial_overflow, sizeof(uint32_t), cudaMemcpyDeviceToHost), "read partial_overflow");

        uint32_t to_copy = std::min(count, PARTIAL_RESULT_CAPACITY);
        if (to_copy != 0) {
            ck(cudaMemcpy(partialHost.data(), gpu.d_partial_results, (size_t)to_copy * sizeof(PartialResult), cudaMemcpyDeviceToHost), "read partial_results");
        }

        uint32_t zeroU = 0u;
        ck(cudaMemcpy(gpu.d_partial_count, &zeroU, sizeof(uint32_t), cudaMemcpyHostToDevice), "reset partial_count");
        ck(cudaMemcpy(gpu.d_partial_overflow, &zeroU, sizeof(uint32_t), cudaMemcpyHostToDevice), "reset partial_overflow");

        for (uint32_t i = 0; i < to_copy; ++i) {
            const PartialResult& r = partialHost[i];
            std::ofstream out(partialOutputFile(partial_digits, r.match_chars), std::ios::app);
            out << "Match: " << r.match_chars << " hex chars\n";
            out << "Hash160: " << formatHash160Hex(r.hash160) << "\n";
            out << "Private Key: " << formatHex256(r.scalar) << "\n";
            out << "Public Key: " << formatCompressedPubHexFromPrefixX(r.pubkey_prefix, r.X) << "\n";
            out << "GPU: " << gpuIdx << "\n\n";
        }

        if (overflow != 0) {
            std::cout << "\nPartial result buffer overflow on GPU " << gpuIdx
                      << ": " << overflow << " candidate(s) were not saved. Increase --partial to reduce hit rate.\n";
        }
    };

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);
        
        unsigned int zeroU = 0u;
        ck(cudaMemcpyAsync(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, gpu.stream), "zero d_any_left");
        
        kernel_point_add_and_check_oneinv<<<gpu.blocks, gpu.threadsPerBlock, 0, gpu.stream>>>(
            gpu.d_Px, gpu.d_Py, gpu.d_Rx, gpu.d_Ry,
            gpu.d_start_scalars, gpu.d_counts256,
            gpu.threadsTotal,
            B,
            slices_per_launch,
            gpu.d_found_flag, gpu.d_found_result,
            gpu.d_partial_results, gpu.d_partial_count, gpu.d_partial_overflow,
            partial_digits, PARTIAL_RESULT_CAPACITY,
            gpu.d_hashes_accum,
            gpu.d_any_left
        );
        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess) {
            std::cerr << "\nKernel launch error on GPU " << gpuIdx << ": " << cudaGetErrorString(launchErr) << "\n";
            return EXIT_FAILURE;
        }
        cudaEventRecord(gpu.kernelDone, gpu.stream);
        gpuNeedsLaunch[gpuIdx] = false;
    }

    while (!stop_all) {
        if (g_sigint) {
            std::cerr << "\n[Ctrl+C] Interrupt received. Finishing current kernel slices and exiting...\n";
            stop_all = true;
        }

        bool any_stream_busy = false;
        unsigned long long totalHashes = 0ull;

        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            if (gpuCompleted[gpuIdx]) {
                totalHashes += gpuHashes[gpuIdx];
                continue;
            }
            GPUContext& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);

            cudaError_t qs = cudaEventQuery(gpu.kernelDone);
            if (qs == cudaSuccess) {
                cudaDeviceSynchronize();
                drain_partial_results(gpu, gpuIdx);
                unsigned int h_any = 0u;
                ck(cudaMemcpy(&h_any, gpu.d_any_left, sizeof(unsigned int), cudaMemcpyDeviceToHost), "read any_left");
                if (h_any == 0u) {
                    gpuCompleted[gpuIdx] = true;
                } else {
                    gpuSlice[gpuIdx]++;
                    if (gpuSlice[gpuIdx] >= slices_per_launch) gpuSlice[gpuIdx] = 0;
                    std::swap(gpu.d_Px, gpu.d_Rx);
                    std::swap(gpu.d_Py, gpu.d_Ry);
                    gpuNeedsLaunch[gpuIdx] = true;
                }
            } else if (qs == cudaErrorNotReady) {
                any_stream_busy = true;
            } else {
                cudaGetLastError();
                stop_all = true;
                std::cerr << "Warning: GPU " << gpuIdx << " stream error\n";
            }

            unsigned long long h_hashes = 0ull;
            cudaMemcpy(&h_hashes, gpu.d_hashes_accum, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            gpuHashes[gpuIdx] = h_hashes;
            totalHashes += h_hashes;

            if (!gpuCompleted[gpuIdx]) {
                std::array<uint64_t,4> curKey;
                uint64_t sampleIdx = gpu.threadsTotal > 1 ? (gpu.threadsTotal / 2) : 0;
                cudaMemcpy(curKey.data(), gpu.d_start_scalars + sampleIdx*4, 4*sizeof(uint64_t), cudaMemcpyDeviceToHost);
                gpuCurrentKey[gpuIdx] = curKey;
            }

            int host_found = 0;
            cudaMemcpy(&host_found, gpu.d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
            if (host_found == FOUND_READY) {
                stop_all = true;
                found_any.store(true);
                int found_gpu = gpuIdx;
                FoundResult host_result{};
                cudaMemcpy(&host_result, gpu.d_found_result, sizeof(FoundResult), cudaMemcpyDeviceToHost);
                cudaDeviceSynchronize();
                drain_partial_results(gpu, gpuIdx);
                std::cout << "\n======== FOUND MATCH! =================================\n";
                std::cout << "Found on GPU " << found_gpu << "\n";
                std::cout << "Private Key   : " << formatHex256(host_result.scalar) << "\n";
                std::cout << "Public Key    : " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
                
                
                std::ofstream out("found_key.txt");
                out << "Private Key: " << formatHex256(host_result.scalar) << "\n";
                out << "Public Key: " << formatCompressedPubHex(host_result.Rx, host_result.Ry) << "\n";
                out << "GPU: " << found_gpu << "\n";
                out.close();
                std::cout << "Result saved to found_key.txt\n";
                break;
            }
        }

        if (stop_all) break;

        for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
            if (!gpuNeedsLaunch[gpuIdx] || gpuCompleted[gpuIdx]) continue;
            GPUContext& gpu = gpus[gpuIdx];
            cudaSetDevice(gpu.deviceId);

            unsigned int zeroU = 0u;
            ck(cudaMemcpyAsync(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, gpu.stream), "zero d_any_left");

            kernel_point_add_and_check_oneinv<<<gpu.blocks, gpu.threadsPerBlock, 0, gpu.stream>>>(
                gpu.d_Px, gpu.d_Py, gpu.d_Rx, gpu.d_Ry,
                gpu.d_start_scalars, gpu.d_counts256,
                gpu.threadsTotal,
                B,
                slices_per_launch,
                gpu.d_found_flag, gpu.d_found_result,
                gpu.d_partial_results, gpu.d_partial_count, gpu.d_partial_overflow,
                partial_digits, PARTIAL_RESULT_CAPACITY,
                gpu.d_hashes_accum,
                gpu.d_any_left
            );
            cudaError_t launchErr = cudaGetLastError();
            if (launchErr != cudaSuccess) {
                std::cerr << "\nKernel launch error on GPU " << gpuIdx << ": " << cudaGetErrorString(launchErr) << "\n";
                stop_all = true;
            }
            cudaEventRecord(gpu.kernelDone, gpu.stream);
            gpuNeedsLaunch[gpuIdx] = false;
        }

        if (!any_stream_busy && !stop_all && !random_mode) {
            bool all_completed = true;
            for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
                if (!gpuCompleted[gpuIdx]) {
                    all_completed = false;
                    break;
                }
            }
            if (all_completed) {
                stop_all = true;
            }
        }

        if (any_stream_busy && !stop_all) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }

        auto now = std::chrono::high_resolution_clock::now();
        double dt = std::chrono::duration<double>(now - tLast).count();
        if (dt >= 1.0) {
            double delta = (double)(totalHashes - lastHashes);
            double mkeys = delta / (dt * 1e6);
            double elapsed = std::chrono::duration<double>(now - t0).count();
            long double total_keys_ld = ld_from_u256(range_len);
            long double prog = total_keys_ld > 0.0L ? ((long double)totalHashes / total_keys_ld) * 100.0L : 0.0L;
            if (prog > 100.0L) prog = 100.0L;

            std::cout << "\rTime: " << std::fixed << std::setprecision(1) << elapsed
                      << " s | Speed: " << std::fixed << std::setprecision(1) << mkeys
                      << " Mkeys/s | Count: " << totalHashes
                      << " | Progress: " << std::fixed << std::setprecision(2) << (double)prog << " %";
            
            for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
                if (gpuCompleted[gpuIdx]) std::cout << " [GPU" << gpuIdx << ":done]";
                else std::cout << " [GPU" << gpuIdx << ":S" << (gpuSlice[gpuIdx]+1) << "/" << slices_per_launch << "|" << formatHex256Trimmed(gpuCurrentKey[gpuIdx].data()) << "]";
            }
            std::cout.flush();
            lastHashes = totalHashes; tLast = now;

            if (random_mode && !stop_all) {
                double elapsed_total = std::chrono::duration<double>(now - t0).count();
                static double last_random_time = 0;
                if (elapsed_total - last_random_time >= random_interval_seconds) {
                    last_random_time = elapsed_total;
                    std::cout << "\n[RANDOM MODE] Re-randomizing keys...\n";

                    uint64_t global_offset[4] = {0, 0, 0, 0};

                    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
                        GPUContext& gpu = gpus[gpuIdx];
                        cudaSetDevice(gpu.deviceId);
                        cudaStreamSynchronize(gpu.stream);
                        drain_partial_results(gpu, gpuIdx);

                        uint64_t cur_offset[4] = { global_offset[0], global_offset[1], global_offset[2], global_offset[3] };
                        for (uint64_t t = 0; t < gpu.threadsTotal; ++t) {
                            uint64_t base_key[4];
                            random_segment_start(base_key, random_sweep_origin, cur_offset, range_start, range_len);

                            uint64_t Sc[4];
                            add256_u64(base_key, half, Sc);
                            cudaMemcpy(gpu.d_start_scalars + t*4, Sc, 4*sizeof(uint64_t), cudaMemcpyHostToDevice);

                            cudaMemcpy(gpu.d_counts256 + t*4, gpu.per_thread_cnt, 4*sizeof(uint64_t), cudaMemcpyHostToDevice);

                            uint64_t next_offset[4];
                            add256(cur_offset, gpu.per_thread_cnt, next_offset);
                            cur_offset[0]=next_offset[0]; cur_offset[1]=next_offset[1];
                            cur_offset[2]=next_offset[2]; cur_offset[3]=next_offset[3];
                        }
                        global_offset[0]=cur_offset[0]; global_offset[1]=cur_offset[1];
                        global_offset[2]=cur_offset[2]; global_offset[3]=cur_offset[3];

                        unsigned int zeroU = 0u;
                        int zero = FOUND_NONE;
                        cudaMemcpy(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice);
                        cudaMemcpy(gpu.d_found_flag, &zero, sizeof(int), cudaMemcpyHostToDevice);

                        int blocks_scal = (int)((gpu.threadsTotal + gpu.threadsPerBlock - 1) / gpu.threadsPerBlock);
                        scalarMulKernelBase<<<blocks_scal, gpu.threadsPerBlock>>>(gpu.d_start_scalars, gpu.d_Px, gpu.d_Py, (int)gpu.threadsTotal);
                        cudaDeviceSynchronize();

                        cudaMemcpyAsync(gpu.d_any_left, &zeroU, sizeof(unsigned int), cudaMemcpyHostToDevice, gpu.stream);
                        kernel_point_add_and_check_oneinv<<<gpu.blocks, gpu.threadsPerBlock, 0, gpu.stream>>>(
                            gpu.d_Px, gpu.d_Py, gpu.d_Rx, gpu.d_Ry,
                            gpu.d_start_scalars, gpu.d_counts256,
                            gpu.threadsTotal,
                            B,
                            slices_per_launch,
                            gpu.d_found_flag, gpu.d_found_result,
                            gpu.d_partial_results, gpu.d_partial_count, gpu.d_partial_overflow,
                            partial_digits, PARTIAL_RESULT_CAPACITY,
                            gpu.d_hashes_accum,
                            gpu.d_any_left
                        );
                        cudaEventRecord(gpu.kernelDone, gpu.stream);
                        gpuSlice[gpuIdx] = 0;
                        gpuCompleted[gpuIdx] = false;
                        gpuNeedsLaunch[gpuIdx] = false;
                    }
                    advance_sweep_origin(random_sweep_origin, random_sweep_coverage, range_start, range_len);
                    std::cout.flush();
                }
            }
        }

        if (!any_stream_busy && !stop_all && !random_mode) {
            bool all_completed = true;
            for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
                if (!gpuCompleted[gpuIdx]) {
                    all_completed = false;
                    break;
                }
            }
            if (all_completed) {
                stop_all = true;
            }
        }
    }

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);
        cudaStreamSynchronize(gpu.stream);
        drain_partial_results(gpu, gpuIdx);
    }
    std::cout << "\n";

    int exit_code = EXIT_SUCCESS;

    if (!found_any.load()) {
        if (g_sigint) {
            std::cout << "======== INTERRUPTED (Ctrl+C) ==========================\n";
            std::cout << "Search was interrupted by user. Partial progress above.\n";
            exit_code = 130;
        } else if (random_mode) {
            std::cout << "======== KEY NOT FOUND (random mode) ==================\n";
            std::cout << "Target hash160 was not found in random sweeps.\n";
        } else {
            std::cout << "======== KEY NOT FOUND (exhaustive) ===================\n";
            std::cout << "Target hash160 was not found within the specified range.\n";
        }
    }

    for (int gpuIdx = 0; gpuIdx < numGPUs; ++gpuIdx) {
        GPUContext& gpu = gpus[gpuIdx];
        cudaSetDevice(gpu.deviceId);
        cudaFree(gpu.d_start_scalars); cudaFree(gpu.d_Px); cudaFree(gpu.d_Py);
        cudaFree(gpu.d_Rx); cudaFree(gpu.d_Ry); cudaFree(gpu.d_counts256);
        cudaFree(gpu.d_found_flag); cudaFree(gpu.d_found_result);
        cudaFree(gpu.d_partial_results); cudaFree(gpu.d_partial_count); cudaFree(gpu.d_partial_overflow);
        cudaFree(gpu.d_hashes_accum); cudaFree(gpu.d_any_left);
        cudaEventDestroy(gpu.kernelDone);
        cudaStreamDestroy(gpu.stream);
    }

    return exit_code;
}
