#ifndef BONUS_IMAGE_PROC_H
#define BONUS_IMAGE_PROC_H

#include "bench.h"
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

const std::chrono::nanoseconds BASELINE_IMAGE_PROC_BONUS{43000000};

struct image_proc_args_bonus {
    std::vector<float> r_channel;
    std::vector<float> g_channel;
    std::vector<float> b_channel;
    std::vector<float> output;

    float threshold;
    size_t width;
    size_t height;
    double epsilon;

    explicit image_proc_args_bonus(double eps = 1e-5)
        : threshold(0.5f), width(0), height(0), epsilon(eps) {}
};

void naive_image_proc_bonus(image_proc_args_bonus& state);
void stu_image_proc_bonus(image_proc_args_bonus& state);

void naive_image_proc_bonus_wrapper(void* ctx);
void stu_image_proc_bonus_wrapper(void* ctx);

void initialize_image_proc_bonus(image_proc_args_bonus* state,
                                 size_t w,
                                 size_t h,
                                 uint64_t seed);
bool image_proc_bonus_check(void* stu_ctx, void* ref_ctx, lab_test_func naive_func);

#endif
