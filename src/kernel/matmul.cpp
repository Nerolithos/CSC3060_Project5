#include "matmul.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

void initialize_matmul(matmul_args& args, int n, uint32_t seed) {
    if (n <= 0) {
        throw std::invalid_argument("initialize_matmul: n must be positive.");
    }

    args.n = n;
    args.epsilon = 1e-3;

    const size_t elem_count = static_cast<size_t>(n) * static_cast<size_t>(n);
    args.A.resize(elem_count);
    args.B.resize(elem_count);
    args.C.assign(elem_count, 0.0f);

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < elem_count; ++i) {
        args.A[i] = dist(rng);
        args.B[i] = dist(rng);
    }
}

void naive_matmul(std::vector<float>& C,
                  const std::vector<float>& A,
                  const std::vector<float>& B,
                  int n) {
    std::fill(C.begin(), C.end(), 0.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < n; ++k) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void stu_matmul(std::vector<float>& C,
                const std::vector<float>& A,
                const std::vector<float>& B,
                int n) {
    constexpr int J_BLOCK = 128;

    std::fill(C.begin(), C.end(), 0.0f);

    const float* __restrict__ a_base = A.data();
    const float* __restrict__ b_base = B.data();
    float* __restrict__ c_base = C.data();

    auto worker = [&](int row_begin, int row_end) {
        for (int i = row_begin; i < row_end; ++i) {
            float* __restrict__ c_row =
                c_base + static_cast<std::size_t>(i) * n;
            const float* __restrict__ a_row =
                a_base + static_cast<std::size_t>(i) * n;

            for (int jj = 0; jj < n; jj += J_BLOCK) {
                const int j_end = std::min(jj + J_BLOCK, n);

                for (int k = 0; k < n; ++k) {
                    const float av = a_row[k];
                    const float* __restrict__ b_row =
                        b_base + static_cast<std::size_t>(k) * n;

                    int j = jj;
                    for (; j + 7 < j_end; j += 8) {
                        c_row[j + 0] += av * b_row[j + 0];
                        c_row[j + 1] += av * b_row[j + 1];
                        c_row[j + 2] += av * b_row[j + 2];
                        c_row[j + 3] += av * b_row[j + 3];
                        c_row[j + 4] += av * b_row[j + 4];
                        c_row[j + 5] += av * b_row[j + 5];
                        c_row[j + 6] += av * b_row[j + 6];
                        c_row[j + 7] += av * b_row[j + 7];
                    }
                    for (; j < j_end; ++j) {
                        c_row[j] += av * b_row[j];
                    }
                }
            }
        }
    };

    const unsigned hw = std::thread::hardware_concurrency();
    const int thread_count =
        std::min<int>(16, std::max<int>(1, hw == 0 ? 8 : static_cast<int>(hw)));

    if (thread_count <= 1 || n < 128) {
        worker(0, n);
        return;
    }

    std::vector<std::thread> threads;
    threads.reserve(static_cast<std::size_t>(thread_count - 1));
    const int rows_per_thread = (n + thread_count - 1) / thread_count;

    int row_begin = 0;
    for (int t = 1; t < thread_count; ++t) {
        const int row_end = std::min(n, row_begin + rows_per_thread);
        if (row_begin >= row_end) {
            break;
        }
        threads.emplace_back(worker, row_begin, row_end);
        row_begin = row_end;
    }

    if (row_begin < n) {
        worker(row_begin, n);
    }

    for (auto& thread : threads) {
        thread.join();
    }
}

void naive_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    naive_matmul(args.C, args.A, args.B, args.n);
}

void stu_matmul_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    stu_matmul(args.C, args.A, args.B, args.n);
}

bool matmul_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func) {
    naive_func(ref_ctx);

    auto& stu_args = *static_cast<matmul_args*>(stu_ctx);
    auto& ref_args = *static_cast<matmul_args*>(ref_ctx);

    if (stu_args.C.size() != ref_args.C.size()) {
        debug_log("\tDEBUG: matmul size mismatch: stu={} ref={}\n",
                  stu_args.C.size(),
                  ref_args.C.size());
        return false;
    }

    const double eps = ref_args.epsilon;
    const int n = ref_args.n;
    double max_rel = 0.0;
    size_t worst_idx = 0;

    for (size_t i = 0; i < ref_args.C.size(); ++i) {
        const double r = static_cast<double>(ref_args.C[i]);
        const double s = static_cast<double>(stu_args.C[i]);
        const double diff = std::abs(s - r);
        const double rel = (std::abs(r) > 1e-9) ? diff / std::abs(r) : diff;

        if (rel > max_rel) {
            max_rel = rel;
            worst_idx = i;
        }

        if (rel > eps) {
            const size_t row = (n > 0) ? (i / static_cast<size_t>(n)) : 0;
            const size_t col = (n > 0) ? (i % static_cast<size_t>(n)) : 0;
            debug_log("\tDEBUG: matmul fail at index {} (row={}, col={}): ref={} stu={} rel={} eps={}\n",
                      i,
                      row,
                      col,
                      ref_args.C[i],
                      stu_args.C[i],
                      rel,
                      eps);
            return false;
        }
    }

    debug_log("\tDEBUG: matmul_check passed. max_rel={} at index {}\n",
              max_rel,
              worst_idx);
    return true;
}
