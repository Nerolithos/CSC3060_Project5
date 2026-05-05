#include "matmul_bonus_openmp.h"

#include <algorithm>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kTileSize = 64;

} // namespace

void bonus_matmul_openmp(std::vector<float>& C,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         int n) {
    std::fill(C.begin(), C.end(), 0.0f);

#pragma omp parallel for collapse(2) schedule(static) if(n >= 128)
    for (int ii = 0; ii < n; ii += kTileSize) {
        for (int jj = 0; jj < n; jj += kTileSize) {
            const int i_end = std::min(ii + kTileSize, n);
            const int j_end = std::min(jj + kTileSize, n);

            for (int kk = 0; kk < n; kk += kTileSize) {
                const int k_end = std::min(kk + kTileSize, n);

                for (int i = ii; i < i_end; ++i) {
                    float* c_row = C.data() + static_cast<std::size_t>(i) * n;
                    const float* a_row =
                        A.data() + static_cast<std::size_t>(i) * n;

                    for (int k = kk; k < k_end; ++k) {
                        const float a_val = a_row[k];
                        const float* b_row =
                            B.data() + static_cast<std::size_t>(k) * n;

                        int j = jj;
                        for (; j + 3 < j_end; j += 4) {
                            c_row[j + 0] += a_val * b_row[j + 0];
                            c_row[j + 1] += a_val * b_row[j + 1];
                            c_row[j + 2] += a_val * b_row[j + 2];
                            c_row[j + 3] += a_val * b_row[j + 3];
                        }
                        for (; j < j_end; ++j) {
                            c_row[j] += a_val * b_row[j];
                        }
                    }
                }
            }
        }
    }
}

void bonus_matmul_openmp_wrapper(void* ctx) {
    auto& args = *static_cast<matmul_args*>(ctx);
    bonus_matmul_openmp(args.C, args.A, args.B, args.n);
}