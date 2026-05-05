#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>

#include "bench.h"
#include "image_proc.h"   // bonus header

static constexpr int RUNS = 20;

int main() {
    std::cout << "Bonus Benchmark\n";

    constexpr std::size_t W = 1024;
    constexpr std::size_t H = 1000;
    constexpr std::uint64_t SEED = 12345u;

    image_proc_args_bonus stu_args;
    image_proc_args_bonus ref_args;
    initialize_image_proc_bonus(&stu_args, W, H, SEED);
    initialize_image_proc_bonus(&ref_args, W, H, SEED);

    std::println("\tImage Proc Bonus: {} x {}", W, H);

    // Correctness check
    bool ok = image_proc_bonus_check(
        static_cast<void*>(&stu_args),
        static_cast<void*>(&ref_args),
        naive_image_proc_bonus_wrapper);
    if (!ok) {
        std::cerr << "FAILED correctness check!\n";
        return 1;
    }
    // Reset output for timing
    std::fill(stu_args.output.begin(), stu_args.output.end(), 0.0f);

    // Warm-up
    stu_image_proc_bonus_wrapper(static_cast<void*>(&stu_args));

    // Timed runs
    long long total_ns = 0;
    for (int i = 0; i < RUNS; ++i) {
        std::fill(stu_args.output.begin(), stu_args.output.end(), 0.0f);
        const auto t0 = std::chrono::steady_clock::now();
        stu_image_proc_bonus_wrapper(static_cast<void*>(&stu_args));
        const auto t1 = std::chrono::steady_clock::now();
        total_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    }
    const long long avg_ns = total_ns / RUNS;
    const double speedup = static_cast<double>(BASELINE_IMAGE_PROC_BONUS.count())
                         / static_cast<double>(avg_ns);

    const char* status = (speedup >= 1.0) ? "PASSED" : "FAILED";
    std::cout << "-------------------------------------------------------\n";
    std::println("Image Proc Bonus  {:>10}   {:>12} ns   {:.3f}x   {}",
                 status, avg_ns, speedup, speedup >= 1.0 ? "" : "<-- below baseline");
    std::cout << "-------------------------------------------------------\n";

    return 0;
}
