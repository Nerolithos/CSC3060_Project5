#include "filter_gradient.h"

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>

void initialize_filter_gradient(filter_gradient_args* args,
                        std::size_t width,
                        std::size_t height,
                        std::uint_fast64_t seed) {
    if (!args) {
        return;
    }

    assert(width >= 3);
    assert(height >= 3);

    args->width = width;
    args->height = height;
    args->out = 0.0f;

    const std::size_t count = width * height;

    std::mt19937_64 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    args->data.a.resize(count);
    args->data.b.resize(count);
    args->data.c.resize(count);
    args->data.d.resize(count);
    args->data.e.resize(count);
    args->data.f.resize(count);
    args->data.g.resize(count);
    args->data.h.resize(count);
    args->data.i.resize(count);

    for (std::size_t k = 0; k < count; ++k) {
        args->data.a[k] = dist(gen);
        args->data.b[k] = dist(gen);
        args->data.c[k] = dist(gen);
        args->data.d[k] = dist(gen);
        args->data.e[k] = dist(gen);
        args->data.f[k] = dist(gen);
        args->data.g[k] = dist(gen);
        args->data.h[k] = dist(gen);
        args->data.i[k] = dist(gen);
    }
}

void naive_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    double total = 0.0f;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        for (std::size_t x = 1; x + 1 < W; ++x) {

            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * W;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += data.a[idx];
                    sum_b += data.b[idx];
                    sum_c += data.c[idx];
                }
            }
            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const std::size_t ym1 = (y - 1) * W;
            const std::size_t y0  = y * W;
            const std::size_t yp1 = (y + 1) * W;

            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;

            const float sobel_dx =
                -data.d[ym1 + xm1] + data.d[ym1 + xp1]
                -2.0f * data.d[y0 + xm1] + 2.0f * data.d[y0 + xp1]
                -data.d[yp1 + xm1] + data.d[yp1 + xp1];

            const float sobel_ex =
                -data.e[ym1 + xm1] + data.e[ym1 + xp1]
                -2.0f * data.e[y0 + xm1] + 2.0f * data.e[y0 + xp1]
                -data.e[yp1 + xm1] + data.e[yp1 + xp1];

            const float sobel_fx =
                -data.f[ym1 + xm1] + data.f[ym1 + xp1]
                -2.0f * data.f[y0 + xm1] + 2.0f * data.f[y0 + xp1]
                -data.f[yp1 + xm1] + data.f[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -data.g[ym1 + xm1] - 2.0f * data.g[ym1 + x0] - data.g[ym1 + xp1]
                + data.g[yp1 + xm1] + 2.0f * data.g[yp1 + x0] + data.g[yp1 + xp1];

            const float sobel_hy =
                -data.h[ym1 + xm1] - 2.0f * data.h[ym1 + x0] - data.h[ym1 + xp1]
                + data.h[yp1 + xm1] + 2.0f * data.h[yp1 + x0] + data.h[yp1 + xp1];

            const float sobel_iy =
                -data.i[ym1 + xm1] - 2.0f * data.i[ym1 + x0] - data.i[ym1 + xp1]
                + data.i[yp1 + xm1] + 2.0f * data.i[yp1 + x0] + data.i[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = total;
}

void stu_filter_gradient(float& out, const data_struct& data,
                   std::size_t width, std::size_t height) {
    // Optimized implementation with reduced index calculations
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;

    // Cache array pointers to avoid repeated dereferencing
    const float* a_ptr = data.a.data();
    const float* b_ptr = data.b.data();
    const float* c_ptr = data.c.data();
    const float* d_ptr = data.d.data();
    const float* e_ptr = data.e.data();
    const float* f_ptr = data.f.data();
    const float* g_ptr = data.g.data();
    const float* h_ptr = data.h.data();
    const float* i_ptr = data.i.data();

    double total = 0.0f;
    const std::size_t stride = W;

    for (std::size_t y = 1; y + 1 < H; ++y) {
        const std::size_t ym1 = (y - 1) * stride;
        const std::size_t y0  = y * stride;
        const std::size_t yp1 = (y + 1) * stride;

        for (std::size_t x = 1; x + 1 < W; ++x) {
            // Compute 3x3 box filter for channels a, b, c (Sobel doesn't use this)
            // Actually, the naive version computes sum and average. Let's keep this optimization simple
            // and focus on avoiding redundant index calculations
            
            const std::size_t xm1 = x - 1;
            const std::size_t x0  = x;
            const std::size_t xp1 = x + 1;
            
            // Box filter on a, b, c
            double sum_a = 0.0, sum_b = 0.0, sum_c = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                const std::size_t row = (y + dy) * stride;
                for (int dx = -1; dx <= 1; ++dx) {
                    const std::size_t idx = row + (x + dx);
                    sum_a += a_ptr[idx];
                    sum_b += b_ptr[idx];
                    sum_c += c_ptr[idx];
                }
            }
            const float avg_a = sum_a * inv9;
            const float avg_b = sum_b * inv9;
            const float avg_c = sum_c * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            // Sobel-X filters on d, e, f (with precomputed row indices)
            const float sobel_dx =
                -d_ptr[ym1 + xm1] + d_ptr[ym1 + xp1]
                -2.0f * d_ptr[y0 + xm1] + 2.0f * d_ptr[y0 + xp1]
                -d_ptr[yp1 + xm1] + d_ptr[yp1 + xp1];

            const float sobel_ex =
                -e_ptr[ym1 + xm1] + e_ptr[ym1 + xp1]
                -2.0f * e_ptr[y0 + xm1] + 2.0f * e_ptr[y0 + xp1]
                -e_ptr[yp1 + xm1] + e_ptr[yp1 + xp1];

            const float sobel_fx =
                -f_ptr[ym1 + xm1] + f_ptr[ym1 + xp1]
                -2.0f * f_ptr[y0 + xm1] + 2.0f * f_ptr[y0 + xp1]
                -f_ptr[yp1 + xm1] + f_ptr[yp1 + xp1];

            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            // Sobel-Y filters on g, h, i
            const float sobel_gy =
                -g_ptr[ym1 + xm1] - 2.0f * g_ptr[ym1 + x0] - g_ptr[ym1 + xp1]
                + g_ptr[yp1 + xm1] + 2.0f * g_ptr[yp1 + x0] + g_ptr[yp1 + xp1];

            const float sobel_hy =
                -h_ptr[ym1 + xm1] - 2.0f * h_ptr[ym1 + x0] - h_ptr[ym1 + xp1]
                + h_ptr[yp1 + xm1] + 2.0f * h_ptr[yp1 + x0] + h_ptr[yp1 + xp1];

            const float sobel_iy =
                -i_ptr[ym1 + xm1] - 2.0f * i_ptr[ym1 + x0] - i_ptr[ym1 + xp1]
                + i_ptr[yp1 + xm1] + 2.0f * i_ptr[yp1 + x0] + i_ptr[yp1 + xp1];

            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            total += p1 + p2 + p3;
        }
    }

    out = total;
}

void naive_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    naive_filter_gradient(args.out, args.data, args.width, args.height);
}
void stu_filter_gradient_wrapper(void* ctx) {
    auto& args = *static_cast<filter_gradient_args*>(ctx);
    args.out = 0.0f;
    stu_filter_gradient(args.out, args.data, args.width, args.height);
}

bool filter_gradient_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    auto& stu_args = *static_cast<filter_gradient_args*>(stu_ctx);
    auto& ref_args = *static_cast<filter_gradient_args*>(ref_ctx);

    ref_args.out = 0.0f;
    naive_func(ref_ctx);

    const auto eps = ref_args.epsilon;
    const double s = static_cast<double>(stu_args.out);
    const double r = static_cast<double>(ref_args.out);
    const double err = std::abs(s - r);
    const double atol = 1e-6;
    const double rel = (std::abs(r) > atol) ? err / std::abs(r) : err;
    debug_log("DEBUG: filter_gradient stu={} ref={} err={} rel={}\n",
              stu_args.out,
              ref_args.out,
              err,
              rel);

    return err <= (atol + eps * std::abs(r));
}
