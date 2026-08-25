#ifndef CUDA_HASH_CUH
#define CUDA_HASH_CUH

#include <cstdint>
#include <cuda_runtime.h>
#include <cstring>

struct MatchResult {
    int found;           
    uint8_t publicKey[33];
    uint8_t sha256[32];
    uint8_t ripemd160[20];
};

__device__ void RIPEMD160_from_SHA256_state(const uint32_t sha_state_be[8], uint8_t ripemd20[20]);
__device__ uint32_t RIPEMD160_diag_from_limbs(const uint64_t x_be_limbs[4]);
__device__ uint32_t SHA256_33_diag_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4]);
__device__ uint32_t getHash160Prefix32_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4]);
__device__ void getHash160_33_from_limbs(uint8_t prefix02_03, const uint64_t x_be_limbs[4], uint8_t out20[20]);
__device__ bool verifyHash160_33_from_limbs_rare(
    uint8_t prefix02_03,
    const uint64_t x_be_limbs[4],
    const uint8_t target20[20]);
#endif 
