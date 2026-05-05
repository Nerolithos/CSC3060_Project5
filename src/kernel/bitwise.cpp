#include "bitwise.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <random>

namespace {

constexpr std::array<std::uint8_t, 256> build_bitwise_lut() {
    std::array<std::uint8_t, 256> lut{};
    for (int i = 0; i < 256; ++i) {
        const auto either = static_cast<std::uint8_t>(i);
        lut[static_cast<std::size_t>(i)] =
            static_cast<std::uint8_t>(0xA5u ^ (either & 0x99u));
    }
    return lut;
}

constexpr auto kBitwiseLut = build_bitwise_lut();

} // namespace

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
    std::size_t n = std::min(result.size(), a.size());
    n = std::min(n, b.size());
    auto *__restrict__ dst = reinterpret_cast<std::uint8_t *>(result.data());
    const auto *__restrict__ pa = reinterpret_cast<const std::uint8_t *>(a.data());
    const auto *__restrict__ pb = reinterpret_cast<const std::uint8_t *>(b.data());

    std::size_t i = 0;
    for (; i + 15 < n; i += 16) {
        dst[i + 0] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 0] | pb[i + 0])];
        dst[i + 1] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 1] | pb[i + 1])];
        dst[i + 2] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 2] | pb[i + 2])];
        dst[i + 3] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 3] | pb[i + 3])];
        dst[i + 4] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 4] | pb[i + 4])];
        dst[i + 5] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 5] | pb[i + 5])];
        dst[i + 6] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 6] | pb[i + 6])];
        dst[i + 7] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 7] | pb[i + 7])];
        dst[i + 8] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 8] | pb[i + 8])];
        dst[i + 9] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 9] | pb[i + 9])];
        dst[i + 10] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 10] | pb[i + 10])];
        dst[i + 11] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 11] | pb[i + 11])];
        dst[i + 12] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 12] | pb[i + 12])];
        dst[i + 13] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 13] | pb[i + 13])];
        dst[i + 14] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 14] | pb[i + 14])];
        dst[i + 15] = kBitwiseLut[static_cast<std::uint8_t>(pa[i + 15] | pb[i + 15])];
    }

    for (; i < n; ++i) {
        const auto either = static_cast<std::uint8_t>(pa[i] | pb[i]);
        dst[i] = kBitwiseLut[either];
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
