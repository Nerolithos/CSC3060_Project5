# CSC3060 Project 5 - Code Optimization

This repository is the starter code for the code optimization assignment. The main job for students is to replace the placeholder `stu_*` implementations in `src/kernel/*.cpp` with faster versions, keep the results correct, and use the benchmark harness to measure the speedup.

## Current Best (Avg.) Performance

| **Benchmark**              | **Status** | **Nanoseconds** | **Speedup** |
| -------------------------- | ---------- | --------------- | ----------- |
| Black-Scholes (Student)    | PASSED     | 4,546,671 ns    | 1.056x      |
| Sparse SpMM (Student)      | PASSED     | 69,340,471 ns   | 1.673x      |
| ReLU (Student)             | PASSED     | 368,993 ns      | 1.491x      |
| Bitwise (Student)          | PASSED     | 244,875 ns      | 1.021x      |
| MatMul (Student)           | PASSED     | 5,372,311 ns    | 16.380x     |
| Trace Replay (Student)     | PASSED     | 1,523,292 ns    | 2.232x      |
| Graph (Student)            | PASSED     | 3,246,231 ns    | 1.540x      |
| GRFF (Student)             | PASSED     | 5,981,555 ns    | 1.421x      |
| Image Proc (Student)       | PASSED     | 32,022,123 ns   | 1.343x      |
| Filter Gradient (Student)  | PASSED     | 20,061,097 ns   | 1.246x      |
| **Geometric mean speedup** |            |                 | **1.833x**  |

Full mark with 20/30 Bonus.



## Repository Layout

```text
CSC3060_Project5_Code_Optimization/
├── CMakeLists.txt          # Build rules for the kernels library and executables
├── README.md               # This guide
├── perfUsage.md            # How to use Linux perf for this project
├── include/                # Shared headers and benchmark interfaces
│   ├── bench.h             # Common benchmark harness and timing/check helpers
│   ├── blackscholes.h      # Benchmark-specific args, baseline, wrappers, checker
│   ├── bitwise.h
│   ├── filter_gradient.h
│   ├── graph.h
│   ├── grff.h
│   ├── image_proc.h
│   ├── matmul.h
│   ├── relu.h
│   ├── sparse_spmm.h
│   └── trace_replay.h
├── src/
│   ├── kernel/             # Naive reference code and student code live here
│   │   ├── blackscholes.cpp
│   │   ├── bitwise.cpp
│   │   ├── filter_gradient.cpp
│   │   ├── graph.cpp
│   │   ├── grff.cpp
│   │   ├── image_proc.cpp
│   │   ├── matmul.cpp
│   │   ├── relu.cpp
│   │   ├── sparse_spmm.cpp
│   │   └── trace_replay.cpp
│   ├── main/
│   │   ├── single_bench.cpp # Run a small benchmark set while tuning/debugging
│   │   └── run_all.cpp      # Run the full suite and report speedups
│   └── res/                 # Images used by perfUsage.md
├── .clang-format           # Optional formatting config
└── .clang-tidy             # Optional lint config
```

What each directory is for:

- `include/`: One header per benchmark. Each header usually defines the argument struct, baseline time, initialization helper, naive implementation declaration, student implementation declaration, wrapper functions, and correctness checker.
- `src/kernel/`: The actual benchmark implementations. This is the directory students will edit most often.
- `src/main/`: The programs that run the benchmarks.
- `src/res/`: Screenshot assets used by `perfUsage.md`. These are not benchmark inputs.

Notes:

- `single_bench.cpp` is the easier place to debug one kernel at a time.
- `run_all.cpp` registers the full benchmark suite and is the better entry point for `perf`.

## How The Benchmark Files Are Organized

Most benchmarks follow the same pattern:

- `include/<name>.h`: declares the benchmark argument struct, `BASELINE_*`, `initialize_*`, `naive_*`, `stu_*`, wrapper functions, and the checker.
- `src/kernel/<name>.cpp`: defines the naive implementation, the placeholder student implementation, wrappers, initialization logic, and result checking.

In other words, students normally edit `src/kernel/<name>.cpp`, and sometimes add fields to the matching `include/<name>.h` argument struct if they need extra precomputed data.

## Baseline Values And Default Input Sizes

The baseline times are defined in the benchmark headers and are used by the benchmark harness when it computes per-benchmark speedup. The table below is sorted by header file name in ascending order. For kernels not yet registered in `run_all.cpp`, the default input size column says so explicitly.

| Header file | Baseline constant | Baseline time (ns) | Default input size in `run_all.cpp` |
| --- | --- | --- | --- |
| `bitwise.h` | `BASELINE_BITWISE` | `250,000` | vector length `1,024,000` |
| `blackscholes.h` | `BASELINE_BLACKSCHOLES` | `4,800,000` | `81,l920` options |
| `filter_gradient.h` | `BASELINE_FILTER_GRADIENT` | `25,000,000` | height x width `1024 x 1024` |
| `graph.h` | `BASELINE_GRAPH` | `5,000,000` | `1,024,000` nodes, average degree `8` |
| `grff.h` | `BASELINE_GRFF` | `8,500,000` | feature size `1,024,000` |
| `image_proc.h` | `BASELINE_IMAGE_PROC` | `43,000,000` | image size `1024 x 1000` |
| `matmul.h` | `BASELINE_MATMUL` | `88,000,000` | matrix size `512 x 512` |
| `relu.h` | `BASELINE_RELU` | `550,000` | vector length `1,024,000` |
| `sparse_spmm.h` | `BASELINE_SPARSE_SPMM` | `116,000,000` | LHS (left hand side) CSR matrix of `2048 x 2048`, the default dense RHS (right hand side) is also `2048 x 2048` |
| `trace_replay.h` | `BASELINE_TRACE_REPLAY` | `3,400,000` | `65,536` records and trace length `1,048,576` |
