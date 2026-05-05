#ifndef BONUS_BITWISE_BONUS_SIMD_H
#define BONUS_BITWISE_BONUS_SIMD_H

#include "../include/bitwise.h"

// Bonus-only SIMD implementation idea using AVX2 intrinsics.
void bonus_bitwise_avx2(std::span<std::int8_t> result,
                        std::span<const std::int8_t> a,
                        std::span<const std::int8_t> b);

void bonus_bitwise_avx2_wrapper(void* ctx);

#endif