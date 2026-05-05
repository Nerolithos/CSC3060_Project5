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

## Removed / Not Kept

- `bonus_notes.cpp` was removed because it was documentation-only code and did
  not provide a runnable optimization.
- Other borderline ideas, such as repeated-context caching or approximate math,
  are described in the report/draft instead of being duplicated here as inactive
  source files.
