#include "matmul_bonus_threaded.h"

#include <algorithm>
#include <thread>

void bonus_matmul_threaded(std::vector<float>& C,
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
