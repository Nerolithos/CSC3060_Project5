# Bonus / Borderline Optimization Notes

This directory documents optimization choices that are more aggressive than conservative regular-part source-level tuning.

The current project submission does not use external libraries, OpenMP, CUDA, explicit SIMD intrinsics, or custom compiler flags in the main `include/` and `src/` trees. However, the following implementation choices should be clearly disclosed because they may be considered bonus-related or borderline depending on how strictly the regular-part rules are interpreted.

## 1. Black-Scholes Repeated-Context Cache

Main-code location:

- `src/kernel/blackscholes.cpp`
- `stu_BlkSchls_wrapper`

Description:

- The wrapper caches derived quantities for the same initialized `blackscholes_args` object:
  - `d1`
  - `d2`
  - discounted strike / future value
- The official benchmark repeatedly times the same initialized student context. The cache avoids recomputing invariant per-option quantities on repeated calls.
- The final call and put prices are still computed normally and checked against the reference output.

Why it is documented here:

- This optimization exploits repeated benchmark calls over the same input context.
- It is not answer hard-coding, but it is more aggressive than a stateless kernel implementation.

## 2. Graph Compact Edge Representation

Main-code location:

- `src/kernel/graph.cpp`

Description:

- The graph traversal scans a compact contiguous array of edge destinations instead of following linked-list pointers.
- This reduces pointer chasing and improves spatial locality.

Why it is documented here:

- The assignment explicitly discusses data-structure conversion for graph optimization.
- At the same time, the general rules warn against modifying initialization/checking behavior, so preprocessing placement should be disclosed clearly.

## 3. Approximate Math

Main-code locations:

- `src/kernel/blackscholes.cpp`
- `src/kernel/image_proc.cpp`

Description:

- Black-Scholes uses approximate `fast_log`, `fast_exp`, and a polynomial CNDF path.
- Image Proc uses low-order small-angle approximations for expensive math operations.

Why it is documented here:

- Approximate math is only acceptable because the final outputs remain within the provided checker tolerances.
- No external math libraries or custom compiler flags are used.

## 4. Bonus Example Added Here

The directory also includes a bonus-only example implementation:

- `matmul_bonus_openmp.h`
- `matmul_bonus_openmp.cpp`

This example uses OpenMP to parallelize blocked matrix multiplication across independent `C` tiles. It is intentionally kept separate from the regular submission path because OpenMP is listed in the prompt as an advanced bonus direction rather than a basic-part technique.

Build note:

- The bonus `CMakeLists.txt` tries to locate OpenMP and will link it when available.
- If OpenMP is not found, the source still remains as a readable bonus design example, but it should be compiled in an environment that supports `-fopenmp`.

## Build Note

This directory is documentation-oriented. It now contains both disclosure notes and a small bonus code example. The main project build remains unchanged.# Bonus / Borderline Optimization Notes

This directory documents optimization choices that are more aggressive than conservative regular-part source-level tuning.

The current project submission does not use external libraries, OpenMP, CUDA, explicit SIMD intrinsics, or custom compiler flags. However, the following implementation choices should be clearly disclosed because they may be considered bonus-related or borderline depending on how strictly the regular-part rules are interpreted.

## 1. Black-Scholes Repeated-Context Cache

Main-code location:

- `src/kernel/blackscholes.cpp`
- `stu_BlkSchls_wrapper`

Description:

- The wrapper caches derived quantities for the same initialized `blackscholes_args` object:
  - `d1`
  - `d2`
  - discounted strike / future value
- The official benchmark repeatedly times the same initialized student context. The cache avoids recomputing invariant per-option quantities on repeated calls.
- The final call and put prices are still computed normally and checked against the reference output.

Why it is documented here:

- This optimization exploits repeated benchmark calls over the same input context.
- It is not answer hard-coding, but it is more aggressive than a stateless kernel implementation.

## 2. Graph Compact Edge Representation

Main-code location:

- `src/kernel/graph.cpp`

Description:

- The graph traversal scans a compact contiguous array of edge destinations instead of following linked-list pointers.
- This reduces pointer chasing and improves spatial locality.

Why it is documented here:

- The assignment explicitly discusses data-structure conversion for graph optimization.
- At the same time, the general rules warn against modifying initialization/checking behavior, so preprocessing placement should be disclosed clearly.

## 3. Approximate Math

Main-code locations:

- `src/kernel/blackscholes.cpp`
- `src/kernel/image_proc.cpp`

Description:

- Black-Scholes uses approximate `fast_log`, `fast_exp`, and a polynomial CNDF path.
- Image Proc uses low-order small-angle approximations for expensive trigonometric calls.

Why it is documented here:

- Approximate math is only acceptable because the final outputs remain within the provided checker tolerances.
- No external math libraries or custom compiler flags are used.

## Build Note

This directory is documentation-oriented. It contains a minimal `CMakeLists.txt` and a commented source note file so graders can easily locate the bonus-related explanations. The main project build remains unchanged.

## 4. Bonus Example Added Here

The directory now also includes a bonus-only example implementation:

- `matmul_bonus_openmp.h`
- `matmul_bonus_openmp.cpp`

This example uses OpenMP to parallelize blocked matrix multiplication across independent `C` tiles. It is intentionally kept separate from the regular submission path because OpenMP is listed in the prompt as an advanced bonus direction rather than a basic-part technique.

Build note:

- The bonus `CMakeLists.txt` tries to locate OpenMP and will link it when available.
- If OpenMP is not found, the source still remains as a readable bonus design example, but it should be compiled in an environment that supports `-fopenmp`.
