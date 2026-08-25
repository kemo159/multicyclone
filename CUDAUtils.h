#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

__host__ __forceinline__ void add256_u64(const uint64_t a[4], uint64_t b, uint64_t out[4]) {
    uint64_t sum = a[0] + b;
    uint64_t carry = (sum < a[0]) ? 1ull : 0ull;
    out[0] = sum;
    for (int i = 1; i < 4; ++i) {
        sum = a[i] + carry;
        carry = (sum < a[i]) ? 1ull : 0ull;
        out[i] = sum;
    }
}

__host__ __forceinline__ void add256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t sum = a[i] + b[i];
        uint64_t carry1 = (sum < a[i]) ? 1ull : 0ull;
        uint64_t sum2 = sum + carry;
        uint64_t carry2 = (sum2 < sum) ? 1ull : 0ull;
        out[i] = sum2;
        carry = carry1 | carry2;
    }
}

__host__ __forceinline__ void mul64_u64(uint64_t a, uint64_t b, uint64_t& lo, uint64_t& hi) {
#if defined(_MSC_VER) && !defined(__clang__)
    lo = _umul128(a, b, &hi);
#else
    __uint128_t p = ( __uint128_t)a * b;
    lo = (uint64_t)p;
    hi = (uint64_t)(p >> 64);
#endif
}

__host__ __forceinline__ void add256_u64_mul(const uint64_t a[4], uint64_t mult, uint64_t out[4]) {
    uint64_t carry = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t lo, hi;
        mul64_u64(a[i], mult, lo, hi);
        uint64_t sum = lo + carry;
        out[i] = sum;
        carry = hi + ((sum < lo) ? 1ull : 0ull);
    }
}

__host__ __forceinline__ void sub256(const uint64_t a[4], const uint64_t b[4], uint64_t out[4]) {
    uint64_t borrow = 0;
    for (int i = 0; i < 4; ++i) {
        uint64_t subtrahend = b[i] + borrow;
        uint64_t borrow_from_add = (subtrahend < b[i]) ? 1ull : 0ull;
        uint64_t diff = a[i] - subtrahend;
        borrow = borrow_from_add | ((a[i] < subtrahend) ? 1ull : 0ull);
        out[i] = diff;
    }
}

__host__ __forceinline__ void inc256(uint64_t a[4], uint64_t inc) {
    uint64_t cur = a[0] + inc;
    uint64_t carry = (cur < a[0]) ? 1ull : 0ull;
    a[0] = cur;
    for (int i = 1; i < 4 && carry; ++i) {
        cur = a[i] + carry;
        carry = (cur < a[i]) ? 1ull : 0ull;
        a[i] = cur;
    }
}

__host__ void divmod_256_by_u64(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t &remainder) {
    for (int i = 0; i < 4; ++i) quotient[i] = 0ull;
    if (divisor == 0ull) {
        remainder = 0ull;
        return;
    }
    uint64_t rem = 0ull;
    uint64_t rem_hi = 0ull;
    for (int i = 3; i >= 0; --i) {
        for (int bit = 63; bit >= 0; --bit) {
            rem_hi = rem >> 63;
            rem = (rem << 1) | ((value[i] >> bit) & 1ull);
            if (rem_hi || rem >= divisor) {
                rem -= divisor;
                quotient[i] |= (1ull << bit);
                rem_hi = 0ull;
            }
        }
    }
    remainder = rem;
}

__host__ void divmod_256_by_u64_array(const uint64_t value[4], uint64_t divisor, uint64_t quotient[4], uint64_t remainder[4]) {
    uint64_t rem = 0;
    divmod_256_by_u64(value, divisor, quotient, rem);
    for (int i = 0; i < 4; ++i) remainder[i] = 0;
    remainder[0] = rem;
}

static inline int hex_digit_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static inline bool parse_hex_u64(const std::string& h, uint64_t& out) {
    uint64_t value = 0;
    if (h.empty() || h.size() > 16) return false;
    for (char c : h) {
        int digit = hex_digit_value(c);
        if (digit < 0) return false;
        value = (value << 4) | (uint64_t)digit;
    }
    out = value;
    return true;
}

bool hexToLE64(const std::string& h_in, uint64_t w[4]) {
    std::string h = h_in;
    if (h.size() >= 2 && (h[0] == '0') && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.size() > 64) return false;
    if (h.size() < 64) h = std::string(64 - h.size(), '0') + h;
    if (h.size() != 64) return false;
    for (int i = 0; i < 4; ++i) {
        std::string part = h.substr(i * 16, 16);
        if (!parse_hex_u64(part, w[3 - i])) return false;
    }
    return true;
}
bool hexToHash160(const std::string& h, uint8_t hash160[20]) {
    if (h.size() != 40) return false;
    for (int i = 0; i < 20; ++i) {
        int hi = hex_digit_value(h[(size_t)i * 2]);
        int lo = hex_digit_value(h[(size_t)i * 2 + 1]);
        if (hi < 0 || lo < 0) return false;
        hash160[i] = (uint8_t)((hi << 4) | lo);
    }
    return true;
}
std::string formatHex256(const uint64_t limbs[4]) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (int i = 3; i >= 0; --i) {
        oss << std::setw(16) << limbs[i];
    }
    return oss.str();
}

std::string formatHex256Trimmed(const uint64_t limbs[4]) {
    std::ostringstream oss;
    oss << std::hex << std::uppercase << std::setfill('0');
    for (int i = 3; i >= 0; --i) {
        oss << std::setw(16) << limbs[i];
    }
    std::string s = oss.str();
    size_t pos = s.find_first_not_of('0');
    if (pos == std::string::npos) return "0";
    return s.substr(pos);
}

__device__ __forceinline__ void inc256_device(uint64_t a[4], uint64_t inc) {
    uint64_t cur = a[0] + inc;
    uint64_t carry = (cur < a[0]) ? 1ull : 0ull;
    a[0] = cur;
    for (int i = 1; i < 4 && carry; ++i) {
        cur = a[i] + carry;
        carry = (cur < a[i]) ? 1ull : 0ull;
        a[i] = cur;
    }
}

static __device__ __forceinline__ uint32_t load_u32_le(const uint8_t* p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}

static __device__ __forceinline__ bool hash160_matches_prefix_then_full(
    const uint8_t* __restrict__ h,       
    const uint8_t* __restrict__ target,  
    const uint32_t target_prefix_le)
{
    if (load_u32_le(h) != target_prefix_le) return false;
#pragma unroll
    for (int k = 4; k < 20; ++k) {
        if (h[k] != target[k]) return false;
    }
    return true;
}

__device__ __forceinline__ bool eq256_u64(const uint64_t a[4], uint64_t b) {
    return (a[0]==b) & (a[1]==0ull) & (a[2]==0ull) & (a[3]==0ull);
}

static __device__ __forceinline__ bool hash160_prefix_equals(
    const uint8_t* __restrict__ h,
    uint32_t target_prefix)
{
    return load_u32_le(h) == target_prefix;
}

// вспомогательная: a (256-бит) >= b (u64)?
__device__ __forceinline__ bool ge256_u64(const uint64_t a[4], uint64_t b) {
    if (a[3] | a[2] | a[1]) return true;  // >= 2^64
    return a[0] >= b;
}

__device__ __forceinline__ void sub256_u64_inplace(uint64_t a[4], uint64_t dec) {
    uint64_t borrow = (a[0] < dec) ? 1ull : 0ull;
    a[0] = a[0] - dec;
#pragma unroll
    for (int i = 1; i < 4; ++i) {
        uint64_t ai = a[i];
        uint64_t bi = borrow;
        a[i] = ai - bi;
        borrow = (ai < bi) ? 1ull : 0ull;
        if (!borrow) break;
    }
}

__device__ __forceinline__ unsigned long long warp_reduce_add_ull(unsigned long long v) {
    unsigned mask = 0xFFFFFFFFu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

static inline std::string human_bytes(double bytes) {
    static const char* u[]={"B","KB","MB","GB","TB","PB"};
    int k=0;
    while(bytes>=1024.0 && k<5){ bytes/=1024.0; ++k; }
    std::ostringstream o; o.setf(std::ios::fixed); o<<std::setprecision(bytes<10?2:1)<<bytes<<" "<<u[k];
    return o.str();
}

static inline long double ld_from_u256(const uint64_t v[4]) {
    return std::ldexp((long double)v[3],192) + std::ldexp((long double)v[2],128) + std::ldexp((long double)v[1],64) + (long double)v[0];
}

static inline std::string formatCompressedPubHex(const uint64_t Rx[4], const uint64_t Ry[4]) {
    uint8_t out[33];
    out[0] = (Ry[0] & 1ULL) ? 0x03 : 0x02;
    int off=1;
    for (int limb=3; limb>=0; --limb) {
        uint64_t v = Rx[limb];
        out[off+0]=(uint8_t)(v>>56); out[off+1]=(uint8_t)(v>>48);
        out[off+2]=(uint8_t)(v>>40); out[off+3]=(uint8_t)(v>>32);
        out[off+4]=(uint8_t)(v>>24); out[off+5]=(uint8_t)(v>>16);
        out[off+6]=(uint8_t)(v>> 8); out[off+7]=(uint8_t)(v>> 0);
        off+=8;
    }
    static const char* hexd="0123456789ABCDEF";
    std::string s; s.resize(66);
    for (int i=0;i<33;++i){ s[2*i]=hexd[(out[i]>>4)&0xF]; s[2*i+1]=hexd[out[i]&0xF]; }
    return s;
}

#endif
