# Bonus Notes

This folder contains optional bonus-only code paths that are not wired into the regular benchmark path.

## OpenMP Image Proc

Files:

- `image_proc.h`
- `image_proc.cpp`

Description:

- `image_proc.h/.cpp` is a copied-and-adapted bonus pair based on the provided reference style, with renamed symbols and equivalent usage updates.
- Its student path keeps OpenMP parallelization in the outer loop using `#pragma omp parallel for schedule(static)` when OpenMP is enabled.
- The regular-path submission does not depend on this code.

Build:

- Configure this folder separately with CMake.
- The targets are built only when `find_package(OpenMP)` succeeds.
