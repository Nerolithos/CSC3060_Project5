#include "image_proc_bonus_omp.h"

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace {

static inline float small_sin(float x) {
    const float x2 = x * x;
    return x * (1.0f - x2 * (1.0f / 6.0f));
}

static inline float small_cos(float x) {
    const float x2 = x * x;
    return 1.0f - 0.5f * x2 + (x2 * x2) * (1.0f / 24.0f);
}

} // namespace

void bonus_image_proc_omp(image_proc_args& args) {
    const std::size_t n = args.width * args.height;
    float* __restrict__ out = args.output.data();
    const float* __restrict__ r = args.r_channel.data();
    const float* __restrict__ g = args.g_channel.data();
    const float* __restrict__ b = args.b_channel.data();
    const float threshold = args.threshold;

#pragma omp parallel for schedule(static) num_threads(16)
    for (std::ptrdiff_t idx = 0; idx < static_cast<std::ptrdiff_t>(n); ++idx) {
        const std::size_t i = static_cast<std::size_t>(idx);

        float r_val = r[i] * 1.05f + 0.02f;
        float g_val = g[i] * 1.05f + 0.02f;
        float b_val = b[i] * 1.05f + 0.02f;
        r_val = std::min(r_val, 1.0f);
        g_val = std::min(g_val, 1.0f);
        b_val = std::min(b_val, 1.0f);

        const float gray = r_val * 0.299f + g_val * 0.587f + b_val * 0.114f;
        const float adjusted =
            std::clamp((gray - 0.05f) / 0.90f, 0.0f, 1.0f);
        const float gray_enhance =
            adjusted * adjusted * (3.0f - 2.0f * adjusted);

        const float intensity = gray_enhance * 1.2f;
        const float half_intensity = intensity * 0.5f;
        const float gain_base =
            std::sqrt(half_intensity * half_intensity + 0.1f);
        const float gain =
            (gain_base > 1.0f) ? (1.0f / gain_base) : (gain_base * 0.95f);
        const float hdr = gray_enhance * gain;
        const float compress_val = hdr / (1.0f + hdr);

        constexpr float p0 = 0.11f, p1 = 0.22f, p2 = 0.33f, p3 = 0.44f;
        constexpr float p4 = 0.55f, p5 = 0.66f, p6 = 0.77f, p7 = 0.88f;
        constexpr float p8 = 0.99f, p9 = 1.01f;
        float mask;
        if (compress_val > threshold) {
            mask = (r_val * p0) + (g_val * p1) - (b_val * p2) + p9;
            mask = (mask > 0.8f) ? (mask * p3) : (mask + p4);
        } else {
            mask = (r_val * p5) - (g_val * p6) + (b_val * p7) - p8;
            mask = (mask < 0.2f) ? (mask + p1) : (mask * p2);
        }

        const float noise =
            small_sin(compress_val * p0) * small_cos(r_val * p1);
        mask = std::clamp(mask * 0.7f + noise * 0.3f, 0.0f, 1.0f);

        static constexpr float lut[] = {0.0f, 0.3f, 1.0f, 0.3f, 0.0f};
        const float scaled = mask * 4.0f;
        const int lut_idx = std::clamp(static_cast<int>(scaled), 0, 4);
        const float frac = scaled - static_cast<float>(lut_idx);
        const float weight =
            (lut_idx < 4)
                ? (lut[lut_idx] * (1.0f - frac) + lut[lut_idx + 1] * frac)
                : lut[4];

        out[i] = std::clamp(compress_val * weight, 0.0f, 1.0f);
    }
}

void bonus_image_proc_omp_wrapper(void* ctx) {
    auto& args = *static_cast<image_proc_args*>(ctx);
    bonus_image_proc_omp(args);
}
