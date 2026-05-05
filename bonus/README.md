# Bonus Optimizations

This directory keeps only bonus code with concrete optimization value. The
regular submission does not depend on these files.

## Retained Bonus Code

### AVX2 Bitwise Prototype

Files:

- `bitwise_bonus_simd.h`
- `bitwise_bonus_simd.cpp`

Purpose:

- Uses AVX2 intrinsics to process 32 int8 lanes per iteration.
- Keeps the scalar tail path for lengths that are not multiples of 32.
- Demonstrates the assignment's advanced SIMD direction without changing the
  regular-part benchmark harness or kernel files.

Build note:

- `bonus/CMakeLists.txt` adds `-mavx2` for this target under Clang/GCC.
- The CMake target is enabled only for x86-family processors. On non-x86
  machines, such as local Apple Silicon, the target is skipped instead of
  failing the bonus build.

### Threaded MatMul Prototype

Files:

- `matmul_bonus_threaded.h`
- `matmul_bonus_threaded.cpp`

Purpose:

- Uses `std::thread` to split output rows across CPU cores.
- Keeps each thread writing a disjoint range of C rows while A/B remain read-only.
- Preserves the same per-element k-loop accumulation order as the regular
  blocked implementation.

Build note:

- This is an advanced/bonus threading prototype. The regular
  `src/kernel/matmul.cpp` remains single-threaded.

### OpenMP Image Proc Prototype

Files:

- `image_proc_bonus_omp.h`
- `image_proc_bonus_omp.cpp`

Purpose:

- Uses OpenMP to split the flattened pixel loop across 16 threads.
- Keeps the regular inlined image pipeline unchanged: gain/shift/limit, gray,
  contrast, HDR compression, mask, and importance weighting.
- Each pixel writes one independent output element, so the parallel loop has no
  cross-thread reduction or write conflict.

Build note:

- `bonus/CMakeLists.txt` uses `find_package(OpenMP QUIET)`.
- On the course server with g++/OpenMP support, this target is compiled with
  OpenMP flags. If OpenMP is not found locally, it still builds as a serial
  reference so the bonus directory remains portable.

## Removed / Not Kept

- `bonus_notes.cpp` was removed because it was documentation-only code and did
  not provide a runnable optimization.
- Other borderline ideas, such as repeated-context caching or approximate math,
  are described in the report/draft instead of being duplicated here as inactive
  source files.
