# Bonus Notes

This folder contains optional bonus-only code paths that are not wired into the regular benchmark path.

## OpenMP Image Proc

Files:

- `image_proc_bonus_openmp.h`
- `image_proc_bonus_openmp.cpp`

Description:

- This variant parallelizes the image-processing loop with OpenMP using `#pragma omp parallel for schedule(static)`.
- It preserves the same per-pixel computation as the regular optimized implementation and is intended only for bonus exploration.
- The regular-path submission does not depend on this code.

Build:

- Configure this folder separately with CMake.
- The target is built only when `find_package(OpenMP)` succeeds.
