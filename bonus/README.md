# Bonus / Borderline Optimization Notes

This directory stores more aggressive optimization ideas for bonus exploration.

## 1. Black-Scholes Repeated-Context Cache

Main-code location:

- `src/kernel/blackscholes.cpp`
- `stu_BlkSchls_wrapper`

Description:

- The wrapper caches derived quantities for one initialized context.
- This can reduce repeated work across benchmark iterations.

## 2. Graph Compact Edge Representation

Main-code location:

- `src/kernel/graph.cpp`

Description:

- Traversal uses compact contiguous edge destinations.
- This reduces pointer chasing and improves locality.

## 3. Approximate Math

Main-code locations:

- `src/kernel/blackscholes.cpp`
- `src/kernel/image_proc.cpp`

Description:

- Uses approximation strategies while staying within checker tolerance.

## 4. Bonus Example in This Folder

Files:

- `bitwise_bonus_simd.h`
- `bitwise_bonus_simd.cpp`

Description:

- This example uses AVX2 intrinsics to process 32 int8 lanes per iteration.
- It is kept in `bonus/` and does not affect the regular part.

Build note:

- `bonus/CMakeLists.txt` adds `-mavx2` for this target under Clang/GCC.
- If target machine has no AVX2 support, treat this as reference bonus code.
