#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <print>
#include <vector>
#include <cmath>

#include "bench.h"
#include "bitwise.h"
#include "blackscholes.h"
#include "filter_gradient.h"
#include "graph.h"
#include "grff.h"
#include "image_proc.h"
#include "matmul.h"
#include "relu.h"
#include "sparse_spmm.h"
#include "trace_replay.h"

#define GEOMETRIC_MEAN 1

int main() {
    std::cout << "Student Optimizations Benchmark\n";
    std::cout << "Benchmark setup\n";

    std::uint32_t seed = 12345u;

    blackscholes_args black_args_stu;
    initialize_blackscholes(black_args_stu, 81920, seed);
    std::cout << "\tBlack-Scholes options: " << black_args_stu.spot_price.size()
              << '\n';

    sparse_spmm_args sparse_args_stu;
    initialize_spmm(sparse_args_stu, 512, 512, -1, {}, seed);
    std::cout << "\tSparse A (CSR): " << sparse_args_stu.csr.rows << " x "
              << sparse_args_stu.csr.cols
              << ", nnz=" << sparse_args_stu.csr.values.size() << '\n';

    constexpr size_t relu_size = 1024000;
    relu_args relu_args_stu;
    initialize_relu(&relu_args_stu, relu_size, seed);
    std::println("\tReLU: vector length={}", relu_size);

    constexpr size_t bitwise_size = 1024000;
    bitwise_args bitwise_args_stu;
    initialize_bitwise(&bitwise_args_stu, bitwise_size, seed);
    std::println("\tBitwise: vector length={}", bitwise_size);

    matmul_args matmul_args_stu;
    initialize_matmul(matmul_args_stu, 512, seed);
    std::cout << "\tMatMul: n=" << matmul_args_stu.n << '\n';

    trace_replay_args trace_args_stu;
    initialize_trace_replay(trace_args_stu, 1 << 16, 1 << 20, seed);
    std::cout << "\tTrace Replay: records=" << trace_args_stu.records.size()
              << ", trace_length=" << trace_args_stu.trace.size() << '\n';

    constexpr std::size_t graph_node_count = 1024000;
    constexpr int graph_avg_degree = 8;
    graph_args graph_args_stu;
    initialize_graph(&graph_args_stu, graph_node_count, graph_avg_degree, seed);
    std::cout << "\tGraph: node_count=" << graph_node_count
              << ", avg_degree=" << graph_avg_degree << '\n';

    constexpr std::size_t grff_size = 1024000;
    grff_args grff_args_stu;
    initialize_grff(&grff_args_stu, grff_size, seed);
    std::cout << "\tGRFF: feature size=" << grff_args_stu.a_features.size()
              << '\n';

    constexpr std::size_t image_width = 1024;
    constexpr std::size_t image_height = 1000;
    image_proc_args image_args_stu;
    initialize_image_proc(&image_args_stu, image_width, image_height, seed);
    std::cout << "\tImage Proc: " << image_args_stu.width << " x "
              << image_args_stu.height << '\n';

    const std::size_t WIDTH = 1024;
    const std::size_t HEIGHT = 1024;
    filter_gradient_args filter_gradient_args_stu;
    initialize_filter_gradient(&filter_gradient_args_stu,
                               WIDTH,
                               HEIGHT,
                               seed);
    std::cout << "\tFilter Gradient: " << HEIGHT << " x " << WIDTH << '\n';

    std::vector<bench_t> benchmarks = {
        {"Black-Scholes (Student)",
         stu_BlkSchls_wrapper,
         naive_BlkSchls_wrapper,
         BlkSchls_check,
         &black_args_stu,
         &black_args_stu,
         BASELINE_BLACKSCHOLES},
        {"Sparse SpMM (Student)",
         stu_sparse_spmm_wrapper,
         naive_sparse_spmm_wrapper,
         sparse_spmm_check,
         &sparse_args_stu,
         &sparse_args_stu,
         BASELINE_SPARSE_SPMM},
        {"ReLU (Student)",
         stu_relu_wrapper,
         naive_relu_wrapper,
         relu_check,
         &relu_args_stu,
         &relu_args_stu,
         BASELINE_RELU},
        {"Bitwise (Student)",
         stu_bitwise_wrapper,
         naive_bitwise_wrapper,
         bitwise_check,
         &bitwise_args_stu,
         &bitwise_args_stu,
         BASELINE_BITWISE},
        {"MatMul (Student)",
         stu_matmul_wrapper,
         naive_matmul_wrapper,
         matmul_check,
         &matmul_args_stu,
         &matmul_args_stu,
         BASELINE_MATMUL},
        {"Trace Replay (Student)",
         stu_trace_replay_wrapper,
         naive_trace_replay_wrapper,
         trace_replay_check,
         &trace_args_stu,
         &trace_args_stu,
         BASELINE_TRACE_REPLAY},
        {"Graph (Student)",
         stu_graph_wrapper,
         naive_graph_wrapper,
         graph_check,
         &graph_args_stu,
         &graph_args_stu,
         BASELINE_GRAPH},
        {"GRFF (Student)",
         stu_grff_wrapper,
         naive_grff_wrapper,
         grff_check,
         &grff_args_stu,
         &grff_args_stu,
         BASELINE_GRFF},
        {"Image Proc (Student)",
         stu_image_proc_wrapper,
         naive_image_proc_wrapper,
         image_proc_check,
         &image_args_stu,
         &image_args_stu,
         BASELINE_IMAGE_PROC},
        {"Filter Gradient (Student)",
         stu_filter_gradient_wrapper,
         naive_filter_gradient_wrapper,
         filter_gradient_check,
         &filter_gradient_args_stu,
         &filter_gradient_args_stu,
         BASELINE_FILTER_GRADIENT}};

    std::cout << "\nRunning Benchmarks...\n";
    std::cout
        << "-----------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(28) << "Benchmark" << std::setw(12)
              << "Status" << std::right << std::setw(15) << "Nanoseconds"
              << std::setw(12) << "Speedup" << '\n';
    std::cout
        << "-----------------------------------------------------------------------\n";

#if GEOMETRIC_MEAN
    std::vector<std::chrono::nanoseconds> measured_times;
    measured_times.reserve(benchmarks.size());
    bool gm_enable = true;
#endif

    constexpr int k_best = 20;
    for (const auto &bench : benchmarks) {
        std::chrono::nanoseconds avg_time{0};

        for (int i = 0; i < k_best; ++i) {
            flush_cache();
            const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });
            avg_time += elapsed;
        }
        avg_time /= static_cast<uint64_t>(k_best);

        const bool correct =
            bench.checkFunc(bench.args, bench.ref_args, bench.naiveFunc);

        std::cout << std::left << std::setw(28) << bench.description;
        if (!correct) {
            std::cout << "\033[1;31mFAILED\033[0m" << std::right
                      << std::setw(15) << "N/A" << std::setw(12) << "N/A"
                      << '\n';
            std::cout
                << "  Error: Results do not match naive implementation!\n";
#if GEOMETRIC_MEAN
            gm_enable = false;
#endif
            continue;
        }

        const double speedup =
            static_cast<double>(bench.baseline_time.count()) /
            static_cast<double>(avg_time.count());

        std::cout << "\033[1;32mPASSED\033[0m" << std::right
                  << std::setw(15) << avg_time.count() << " ns"
                  << std::setw(12) << std::fixed << std::setprecision(3)
                  << speedup << "x\n";

#if GEOMETRIC_MEAN
        measured_times.push_back(avg_time);
#endif
    }

#if GEOMETRIC_MEAN
    if (gm_enable && !measured_times.empty()) {
        double gm = 1.0;
        for (const auto &t : measured_times) {
            gm *= static_cast<double>(t.count());
        }
        gm = std::pow(gm, 1.0 / static_cast<double>(measured_times.size()));

        // Convert back to speedup
        // gm is the geometric mean of measured times
        // We need geometric mean of speedups
        std::vector<double> speedups;
        speedups.reserve(benchmarks.size());
        
        // Re-run to get all speedups
        size_t idx = 0;
        for (const auto &bench : benchmarks) {
            std::chrono::nanoseconds avg_time{0};
            for (int i = 0; i < k_best; ++i) {
                flush_cache();
                const auto elapsed = measure_time([&] { bench.tfunc(bench.args); });
                avg_time += elapsed;
            }
            avg_time /= static_cast<uint64_t>(k_best);
            
            const double speedup =
                static_cast<double>(bench.baseline_time.count()) /
                static_cast<double>(avg_time.count());
            speedups.push_back(speedup);
            idx++;
        }
        
        double gm_speedup = 1.0;
        for (double s : speedups) {
            gm_speedup *= s;
        }
        gm_speedup = std::pow(gm_speedup, 1.0 / static_cast<double>(speedups.size()));

        std::cout << "-----------------------------------------------------------------------\n";
        std::cout << "Geometric mean speedup: " << std::fixed << std::setprecision(3)
                  << gm_speedup << "x\n";
    }
#endif

    return 0;
}
