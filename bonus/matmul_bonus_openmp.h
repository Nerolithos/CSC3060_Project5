#ifndef BONUS_MATMUL_BONUS_OPENMP_H
#define BONUS_MATMUL_BONUS_OPENMP_H

#include "../include/matmul.h"

// Bonus-only implementation idea:
// parallelize blocked matmul with OpenMP over independent C tiles.
void bonus_matmul_openmp(std::vector<float>& C,
                         const std::vector<float>& A,
                         const std::vector<float>& B,
                         int n);

void bonus_matmul_openmp_wrapper(void* ctx);

#endif