#include "bitwise_bonus_simd.h"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>

void bonus_bitwise_avx2(std::span<std::int8_t> result,
                        std::span<const std::int8_t> a,
                        std::span<const std::int8_t> b) {
    const std::size_t n = std::min({result.size(), a.size(), b.size()});

    const __m256i kBase = _mm256_set1_epi8(static_cast<char>(0xA5));
    const __m256i kDelta = _mm256_set1_epi8(static_cast<char>(0x99));

    std::size_t i = 0;
    for (; i + 31 < n; i += 32) {
        const __m256i ua = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a.data() + i));
        const __m256i ub = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b.data() + i));

        const __m256i either = _mm256_or_si256(ua, ub);
        const __m256i out = _mm256_xor_si256(kBase,
                                             _mm256_and_si256(either, kDelta));
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), out);
    }

    constexpr std::uint8_t kBaseScalar = 0xA5u;
    constexpr std::uint8_t kDeltaScalar = 0x99u;
    for (; i < n; ++i) {
        const auto either = static_cast<std::uint8_t>(a[i] | b[i]);
        result[i] = static_cast<std::int8_t>(kBaseScalar ^ (either & kDeltaScalar));
    }
}

void bonus_bitwise_avx2_wrapper(void* ctx) {
    auto& args = *static_cast<bitwise_args*>(ctx);
    bonus_bitwise_avx2(args.result, args.a, args.b);
}