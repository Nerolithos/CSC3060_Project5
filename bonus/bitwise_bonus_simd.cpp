#include "bitwise_bonus_simd.h"

#include <algorithm>
#include <cstdint>
#include <immintrin.h>

void bonus_bitwise_avx2(std::span<std::int8_t> result,
                        std::span<const std::int8_t> a,
                        std::span<const std::int8_t> b) {
    const std::size_t n = std::min({result.size(), a.size(), b.size()});

    const __m256i kMaskLo = _mm256_set1_epi8(static_cast<char>(0x5A));
    const __m256i kMaskHi = _mm256_set1_epi8(static_cast<char>(0xC3));
    const __m256i kOnes = _mm256_set1_epi8(static_cast<char>(0xFF));

    std::size_t i = 0;
    for (; i + 31 < n; i += 32) {
        const __m256i ua = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(a.data() + i));
        const __m256i ub = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(b.data() + i));

        const __m256i shared = _mm256_and_si256(ua, ub);
        const __m256i either = _mm256_or_si256(ua, ub);
        const __m256i diff = _mm256_xor_si256(ua, ub);

        const __m256i mixed0 = _mm256_or_si256(
            _mm256_and_si256(diff, kMaskLo),
            _mm256_and_si256(_mm256_xor_si256(shared, kOnes),
                             _mm256_xor_si256(kMaskLo, kOnes)));

        const __m256i mixed1 = _mm256_xor_si256(
            _mm256_and_si256(_mm256_xor_si256(either, kMaskHi),
                             _mm256_or_si256(shared,
                                             _mm256_xor_si256(kMaskHi, kOnes))),
            diff);

        const __m256i out = _mm256_xor_si256(mixed0, mixed1);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(result.data() + i), out);
    }

    constexpr std::uint8_t kMaskLoScalar = 0x5Au;
    constexpr std::uint8_t kMaskHiScalar = 0xC3u;
    for (; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);
        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 = static_cast<std::uint8_t>(
            (diff & kMaskLoScalar) | (~shared & ~kMaskLoScalar));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHiScalar) & (shared | ~kMaskHiScalar)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

void bonus_bitwise_avx2_wrapper(void* ctx) {
    auto& args = *static_cast<bitwise_args*>(ctx);
    bonus_bitwise_avx2(args.result, args.a, args.b);
}