#include "bitwise.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>

void initialize_bitwise(bitwise_args *args, const size_t size,
                                  const std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    constexpr std::int8_t LOWER_BOUND = std::numeric_limits<std::int8_t>::min();
    constexpr std::int8_t UPPER_BOUND = std::numeric_limits<std::int8_t>::max();

    std::mt19937_64 gen(seed);
    std::uniform_int_distribution<int> dist(LOWER_BOUND, UPPER_BOUND);

    args->a.resize(size);
    args->b.resize(size);
    args->result.resize(size);

    for (std::size_t i = 0; i < size; ++i) {
        args->a[i] = static_cast<std::int8_t>(dist(gen));
        args->b[i] = static_cast<std::int8_t>(dist(gen));
        args->result[i] = 0;
    }
}


// The reference implementation of bitwise
// Student should not change this function
void naive_bitwise(std::span<std::int8_t> result,
                   std::span<const std::int8_t> a,
                   std::span<const std::int8_t> b) {
    constexpr std::uint8_t kMaskLo = 0x5Au;
    constexpr std::uint8_t kMaskHi = 0xC3u;

    const std::size_t n = std::min({result.size(), a.size(), b.size()});
    for (std::size_t i = 0; i < n; ++i) {
        const auto ua = static_cast<std::uint8_t>(a[i]);
        const auto ub = static_cast<std::uint8_t>(b[i]);

        const auto shared = static_cast<std::uint8_t>(ua & ub);
        const auto either = static_cast<std::uint8_t>(ua | ub);
        const auto diff = static_cast<std::uint8_t>(ua ^ ub);
        const auto mixed0 =
            static_cast<std::uint8_t>((diff & kMaskLo) | (~shared & ~kMaskLo));
        const auto mixed1 = static_cast<std::uint8_t>(
            ((either ^ kMaskHi) & (shared | ~kMaskHi)) ^ diff);

        result[i] = static_cast<std::int8_t>(mixed0 ^ mixed1);
    }
}

// TODO: Optimize the bitwise function
void stu_bitwise(std::span<std::int8_t> result, std::span<const std::int8_t> a,
                 std::span<const std::int8_t> b) {
    constexpr std::uint8_t kBaseByte = 0xA5u;
    constexpr std::uint8_t kDeltaByte = 0x99u;
    constexpr std::uint64_t kBase64 = 0xA5A5A5A5A5A5A5A5ull;
    constexpr std::uint64_t kDelta64 = 0x9999999999999999ull;

    std::size_t n = std::min(result.size(), a.size());
    n = std::min(n, b.size());
    auto *__restrict__ dst = reinterpret_cast<std::uint8_t *>(result.data());
    const auto *__restrict__ pa = reinterpret_cast<const std::uint8_t *>(a.data());
    const auto *__restrict__ pb = reinterpret_cast<const std::uint8_t *>(b.data());

    std::size_t i = 0;
    const bool aligned64 =
        ((reinterpret_cast<std::uintptr_t>(dst) |
          reinterpret_cast<std::uintptr_t>(pa) |
          reinterpret_cast<std::uintptr_t>(pb)) &
         0x7u) == 0u;

    if (aligned64) {
        auto *__restrict__ dst64 = reinterpret_cast<std::uint64_t *>(dst);
        const auto *__restrict__ pa64 = reinterpret_cast<const std::uint64_t *>(pa);
        const auto *__restrict__ pb64 = reinterpret_cast<const std::uint64_t *>(pb);
        const std::size_t words = n >> 3;
        std::size_t w = 0;
        for (; w + 3 < words; w += 4) {
            dst64[w + 0] = kBase64 ^ ((pa64[w + 0] | pb64[w + 0]) & kDelta64);
            dst64[w + 1] = kBase64 ^ ((pa64[w + 1] | pb64[w + 1]) & kDelta64);
            dst64[w + 2] = kBase64 ^ ((pa64[w + 2] | pb64[w + 2]) & kDelta64);
            dst64[w + 3] = kBase64 ^ ((pa64[w + 3] | pb64[w + 3]) & kDelta64);
        }
        for (; w < words; ++w) {
            dst64[w] = kBase64 ^ ((pa64[w] | pb64[w]) & kDelta64);
        }
        i = words << 3;
    } else {
        for (; i + 32 <= n; i += 32) {
            std::uint64_t ua0;
            std::uint64_t ub0;
            std::uint64_t ua1;
            std::uint64_t ub1;
            std::uint64_t ua2;
            std::uint64_t ub2;
            std::uint64_t ua3;
            std::uint64_t ub3;

            std::memcpy(&ua0, pa + i, 8);
            std::memcpy(&ub0, pb + i, 8);
            std::memcpy(&ua1, pa + i + 8, 8);
            std::memcpy(&ub1, pb + i + 8, 8);
            std::memcpy(&ua2, pa + i + 16, 8);
            std::memcpy(&ub2, pb + i + 16, 8);
            std::memcpy(&ua3, pa + i + 24, 8);
            std::memcpy(&ub3, pb + i + 24, 8);

            const std::uint64_t out0 = kBase64 ^ ((ua0 | ub0) & kDelta64);
            const std::uint64_t out1 = kBase64 ^ ((ua1 | ub1) & kDelta64);
            const std::uint64_t out2 = kBase64 ^ ((ua2 | ub2) & kDelta64);
            const std::uint64_t out3 = kBase64 ^ ((ua3 | ub3) & kDelta64);

            std::memcpy(dst + i, &out0, 8);
            std::memcpy(dst + i + 8, &out1, 8);
            std::memcpy(dst + i + 16, &out2, 8);
            std::memcpy(dst + i + 24, &out3, 8);
        }

        for (; i + 8 <= n; i += 8) {
            std::uint64_t ua;
            std::uint64_t ub;
            std::memcpy(&ua, pa + i, 8);
            std::memcpy(&ub, pb + i, 8);
            const std::uint64_t out = kBase64 ^ ((ua | ub) & kDelta64);
            std::memcpy(dst + i, &out, 8);
        }
    }

    for (; i < n; ++i) {
        const auto either = static_cast<std::uint8_t>(pa[i] | pb[i]);
        dst[i] = static_cast<std::uint8_t>(kBaseByte ^ (either & kDeltaByte));
    }
}

void naive_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    naive_bitwise(args.result, args.a, args.b);
}

void stu_bitwise_wrapper(void *ctx) {
    auto &args = *static_cast<bitwise_args *>(ctx);
    stu_bitwise(args.result, args.a, args.b);
}

bool bitwise_check(void *stu_ctx, void *ref_ctx, lab_test_func naive_func) {
    // Compute reference
    naive_func(ref_ctx);

    auto &stu_args = *static_cast<bitwise_args *>(stu_ctx);
    auto &ref_args = *static_cast<bitwise_args *>(ref_ctx);

    if (stu_args.result.size() != ref_args.result.size()) {
        debug_log("\tDEBUG: size mismatch: stu={} ref={}\n",
                  stu_args.result.size(),
                  ref_args.result.size());
        return false;
    }

    std::int32_t max_abs_diff = 0;
    size_t worst_i = 0;

    for (size_t i = 0; i < ref_args.result.size(); ++i) {
        const auto r = static_cast<std::int32_t>(ref_args.result[i]);
        const auto s = static_cast<std::int32_t>(stu_args.result[i]);

        if (r != s) {
            max_abs_diff = std::abs(r - s);
            worst_i = i;

            debug_log("\tDEBUG: fail at {}: ref={} stu={} abs_diff={}\n",
                      i,
                      r,
                      s,
                      max_abs_diff);
            return false;
        }
    }

    debug_log("\tDEBUG: bitwise_check passed. max_abs_diff={} at i={}\n",
              max_abs_diff,
              worst_i);
    return true;
}
