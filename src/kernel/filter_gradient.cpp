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
    const std::size_t W = width;
    const std::size_t H = height;
    constexpr float inv9 = 1.0f / 9.0f;
    const float *__restrict__ a_data = data.a.data();
    const float *__restrict__ b_data = data.b.data();
    const float *__restrict__ c_data = data.c.data();
    const float *__restrict__ d_data = data.d.data();
    const float *__restrict__ e_data = data.e.data();
    const float *__restrict__ f_data = data.f.data();
    const float *__restrict__ g_data = data.g.data();
    const float *__restrict__ h_data = data.h.data();
    const float *__restrict__ i_data = data.i.data();

    double total = 0.0;
    for (std::size_t y = 1; y + 1 < H; ++y) {
        double row_total = 0.0;
        const std::size_t ym1 = (y - 1) * W;
        const std::size_t y0 = y * W;
        const std::size_t yp1 = (y + 1) * W;

        const float *__restrict__ a0 = a_data + ym1;
        const float *__restrict__ a1 = a_data + y0;
        const float *__restrict__ a2 = a_data + yp1;
        const float *__restrict__ b0 = b_data + ym1;
        const float *__restrict__ b1 = b_data + y0;
        const float *__restrict__ b2 = b_data + yp1;
        const float *__restrict__ c0 = c_data + ym1;
        const float *__restrict__ c1 = c_data + y0;
        const float *__restrict__ c2 = c_data + yp1;
        const float *__restrict__ d0 = d_data + ym1;
        const float *__restrict__ d1 = d_data + y0;
        const float *__restrict__ d2 = d_data + yp1;
        const float *__restrict__ e0 = e_data + ym1;
        const float *__restrict__ e1 = e_data + y0;
        const float *__restrict__ e2 = e_data + yp1;
        const float *__restrict__ f0 = f_data + ym1;
        const float *__restrict__ f1 = f_data + y0;
        const float *__restrict__ f2 = f_data + yp1;
        const float *__restrict__ g0 = g_data + ym1;
        const float *__restrict__ g2 = g_data + yp1;
        const float *__restrict__ h0 = h_data + ym1;
        const float *__restrict__ h2 = h_data + yp1;
        const float *__restrict__ i0 = i_data + ym1;
        const float *__restrict__ i2 = i_data + yp1;

        float a_col0 = a0[0] + a1[0] + a2[0];
        float a_col1 = a0[1] + a1[1] + a2[1];
        float a_col2 = a0[2] + a1[2] + a2[2];
        float b_col0 = b0[0] + b1[0] + b2[0];
        float b_col1 = b0[1] + b1[1] + b2[1];
        float b_col2 = b0[2] + b1[2] + b2[2];
        float c_col0 = c0[0] + c1[0] + c2[0];
        float c_col1 = c0[1] + c1[1] + c2[1];
        float c_col2 = c0[2] + c1[2] + c2[2];

        for (std::size_t x = 1; x + 2 < W; ++x) {
            const std::size_t xm1 = x - 1;
            const std::size_t xp1 = x + 1;

            const float avg_a = (a_col0 + a_col1 + a_col2) * inv9;
            const float avg_b = (b_col0 + b_col1 + b_col2) * inv9;
            const float avg_c = (c_col0 + c_col1 + c_col2) * inv9;
            const float p1 = avg_a * avg_b + avg_c;

            const float sobel_dx =
                -d0[xm1] + d0[xp1]
                -2.0f * d1[xm1] + 2.0f * d1[xp1]
                -d2[xm1] + d2[xp1];
            const float sobel_ex =
                -e0[xm1] + e0[xp1]
                -2.0f * e1[xm1] + 2.0f * e1[xp1]
                -e2[xm1] + e2[xp1];
            const float sobel_fx =
                -f0[xm1] + f0[xp1]
                -2.0f * f1[xm1] + 2.0f * f1[xp1]
                -f2[xm1] + f2[xp1];
            const float p2 = sobel_dx * sobel_ex + sobel_fx;

            const float sobel_gy =
                -g0[xm1] - 2.0f * g0[x] - g0[xp1]
                + g2[xm1] + 2.0f * g2[x] + g2[xp1];
            const float sobel_hy =
                -h0[xm1] - 2.0f * h0[x] - h0[xp1]
                + h2[xm1] + 2.0f * h2[x] + h2[xp1];
            const float sobel_iy =
                -i0[xm1] - 2.0f * i0[x] - i0[xp1]
                + i2[xm1] + 2.0f * i2[x] + i2[xp1];
            const float p3 = sobel_gy * sobel_hy + sobel_iy;

            row_total += p1 + p2 + p3;

            a_col0 = a_col1;
            a_col1 = a_col2;
            a_col2 = a0[x + 2] + a1[x + 2] + a2[x + 2];
            b_col0 = b_col1;
            b_col1 = b_col2;
            b_col2 = b0[x + 2] + b1[x + 2] + b2[x + 2];
            c_col0 = c_col1;
            c_col1 = c_col2;
            c_col2 = c0[x + 2] + c1[x + 2] + c2[x + 2];
        }

        const std::size_t x = W - 2;
        const std::size_t xm1 = x - 1;
        const std::size_t xp1 = x + 1;

        const float avg_a = (a_col0 + a_col1 + a_col2) * inv9;
        const float avg_b = (b_col0 + b_col1 + b_col2) * inv9;
        const float avg_c = (c_col0 + c_col1 + c_col2) * inv9;
        const float p1 = avg_a * avg_b + avg_c;

        const float sobel_dx =
            -d0[xm1] + d0[xp1]
            -2.0f * d1[xm1] + 2.0f * d1[xp1]
            -d2[xm1] + d2[xp1];
        const float sobel_ex =
            -e0[xm1] + e0[xp1]
            -2.0f * e1[xm1] + 2.0f * e1[xp1]
            -e2[xm1] + e2[xp1];
        const float sobel_fx =
            -f0[xm1] + f0[xp1]
            -2.0f * f1[xm1] + 2.0f * f1[xp1]
            -f2[xm1] + f2[xp1];
        const float p2 = sobel_dx * sobel_ex + sobel_fx;

        const float sobel_gy =
            -g0[xm1] - 2.0f * g0[x] - g0[xp1]
            + g2[xm1] + 2.0f * g2[x] + g2[xp1];
        const float sobel_hy =
            -h0[xm1] - 2.0f * h0[x] - h0[xp1]
            + h2[xm1] + 2.0f * h2[x] + h2[xp1];
        const float sobel_iy =
            -i0[xm1] - 2.0f * i0[x] - i0[xp1]
            + i2[xm1] + 2.0f * i2[x] + i2[xp1];
        const float p3 = sobel_gy * sobel_hy + sobel_iy;

        row_total += p1 + p2 + p3;
        total += row_total;
    }

    out = static_cast<float>(total);
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
