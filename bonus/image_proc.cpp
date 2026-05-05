#include "image_proc.h"

#include <algorithm>
#include <cmath>
#include <random>

void initialize_image_proc_bonus(image_proc_args_bonus* state,
                                 size_t w,
                                 size_t h,
                                 uint64_t seed) {
    state->width = w;
    state->height = h;
    const size_t pixel_count = w * h;

    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    state->r_channel.assign(pixel_count, 0.0f);
    state->g_channel.assign(pixel_count, 0.0f);
    state->b_channel.assign(pixel_count, 0.0f);
    state->output.assign(pixel_count, 0.0f);

    for (size_t idx = 0; idx < pixel_count; ++idx) {
        state->r_channel[idx] = dist(rng);
        state->g_channel[idx] = dist(rng);
        state->b_channel[idx] = dist(rng);
    }
}

__attribute__((noinline))
float apply_gain(float v) {
    return v * 1.05f;
}

__attribute__((noinline))
float apply_shift(float v) {
    return v + 0.02f;
}

__attribute__((noinline))
float apply_limit(float v) {
    return (v > 1.0f) ? 1.0f : v;
}

__attribute__((noinline))
float color_correct(float v) {
    return apply_limit(apply_shift(apply_gain(v)));
}

__attribute__((noinline))
float compute_gray(float r, float g, float b) {
    return (r * 0.299f) + (g * 0.587f) + (b * 0.114f);
}

__attribute__((noinline))
float enhance_contrast(float gray) {
    const float adjusted = std::clamp((gray - 0.05f) / 0.90f, 0.0f, 1.0f);
    return adjusted * adjusted * (3.0f - 2.0f * adjusted);
}

__attribute__((noinline))
float calculate_gain(float intensity) {
    const float v1 = intensity * 0.5f;
    const float v2 = v1 * v1 + 0.1f;
    const float v3 = std::sqrt(v2);
    return (v3 > 1.0f) ? (1.0f / v3) : (v3 * 0.95f);
}

__attribute__((noinline))
float hdr_compress(float val) {
    const float gain = calculate_gain(val * 1.2f);
    const float mixed = val * gain;
    return mixed / (1.0f + mixed);
}

__attribute__((noinline))
float complex_mask_logic(float gray, float r, float g, float b, float thresh) {
    constexpr float p0 = 0.11f, p1 = 0.22f, p2 = 0.33f, p3 = 0.44f, p4 = 0.55f;
    constexpr float p5 = 0.66f, p6 = 0.77f, p7 = 0.88f, p8 = 0.99f, p9 = 1.01f;

    float mask = 0.0f;
    if (gray > thresh) {
        mask = (r * p0) + (g * p1) - (b * p2) + p9;
        mask = (mask > 0.8f) ? (mask * p3) : (mask + p4);
    } else {
        mask = (r * p5) - (g * p6) + (b * p7) - p8;
        mask = (mask < 0.2f) ? (mask + p1) : (mask * p2);
    }

    const float noise = std::sin(gray * p0) * std::cos(r * p1);
    const float merged = (mask * 0.7f) + (noise * 0.3f);
    return std::clamp(merged, 0.0f, 1.0f);
}

__attribute__((noinline))
float importance_weight(float val) {
    static const float lut[] = {0.0f, 0.3f, 1.0f, 0.3f, 0.0f};
    const float scaled = val * 4.0f;
    const int lut_idx = std::clamp(static_cast<int>(scaled), 0, 4);
    const float frac = scaled - static_cast<float>(lut_idx);
    if (lut_idx < 4) {
        return lut[lut_idx] * (1.0f - frac) + lut[lut_idx + 1] * frac;
    }
    return lut[4];
}

__attribute__((noinline))
float fast_activate(float val) {
    return val / (1.0f + std::abs(val));
}

void naive_image_proc_bonus(image_proc_args_bonus& state) {
    const size_t w = state.width;
    const size_t h = state.height;
    float* __restrict__ out = state.output.data();
    const float* __restrict__ r = state.r_channel.data();
    const float* __restrict__ g = state.g_channel.data();
    const float* __restrict__ b = state.b_channel.data();
    const float threshold = state.threshold;

    for (size_t y = 0; y < h; ++y) {
        for (size_t x = 0; x < w; ++x) {
            const size_t idx = y * w + x;
            const float r_val = color_correct(r[idx]);
            const float g_val = color_correct(g[idx]);
            const float b_val = color_correct(b[idx]);
            const float gray = compute_gray(r_val, g_val, b_val);
            const float gray_enhance = enhance_contrast(gray);
            const float compress_val = hdr_compress(gray_enhance);
            const float mask =
                complex_mask_logic(compress_val, r_val, g_val, b_val, threshold);
            const float weight = importance_weight(mask);
            out[idx] = std::clamp(compress_val * weight, 0.0f, 1.0f);
        }
    }
}

void stu_image_proc_bonus(image_proc_args_bonus& state) {
    const size_t w = state.width;
    const size_t h = state.height;
    float* __restrict__ out = state.output.data();
    const float* __restrict__ r = state.r_channel.data();
    const float* __restrict__ g = state.g_channel.data();
    const float* __restrict__ b = state.b_channel.data();
    const float threshold = state.threshold;

#if defined(_OPENMP)
#pragma omp parallel for schedule(static)
#endif
    for (std::ptrdiff_t y = 0; y < static_cast<std::ptrdiff_t>(h); ++y) {
        for (size_t x = 0; x < w; ++x) {
            const size_t idx = static_cast<size_t>(y) * w + x;

            const float r_val = color_correct(r[idx]);
            const float g_val = color_correct(g[idx]);
            const float b_val = color_correct(b[idx]);

            const float gray = compute_gray(r_val, g_val, b_val);
            const float gray_enhance = enhance_contrast(gray);
            const float compress_val = hdr_compress(gray_enhance);
            const float mask =
                complex_mask_logic(compress_val, r_val, g_val, b_val, threshold);
            const float weight = importance_weight(mask);

            out[idx] = std::clamp(compress_val * weight, 0.0f, 1.0f);
        }
    }
}

void naive_image_proc_bonus_wrapper(void* ctx) {
    auto& state = *static_cast<image_proc_args_bonus*>(ctx);
    naive_image_proc_bonus(state);
}

void stu_image_proc_bonus_wrapper(void* ctx) {
    auto& state = *static_cast<image_proc_args_bonus*>(ctx);
    stu_image_proc_bonus(state);
}

bool image_proc_bonus_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);
    auto& stu = *static_cast<image_proc_args_bonus*>(stu_ctx);
    auto& ref = *static_cast<image_proc_args_bonus*>(ref_ctx);

    if (stu.output.size() != ref.output.size()) {
        debug_log("DEBUG: image_proc_bonus size mismatch: stu={} ref={}\n",
                  stu.output.size(),
                  ref.output.size());
        return false;
    }

    for (size_t idx = 0; idx < ref.output.size(); ++idx) {
        const double err =
            std::abs(static_cast<double>(stu.output[idx]) -
                     static_cast<double>(ref.output[idx]));
        if (err > 1e-4) {
            debug_log("DEBUG: image_proc_bonus fail at {}: ref={} stu={} err={}\n",
                      idx,
                      ref.output[idx],
                      stu.output[idx],
                      err);
            return false;
        }
    }
    debug_log("DEBUG: image_proc_bonus_check passed. size={}\n", ref.output.size());
    return true;
}
