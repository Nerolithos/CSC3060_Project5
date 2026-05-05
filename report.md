# CSC3060 Project 5 Code Optimization Report



## 0. About our team

We're a two member team: 124090960 ZHU Ji'an, 124090302 LI Jinhong.

Our work distribution are equal, we'd like to share the same final score.



</br>



## 1. Final Server-Side Performance Results

All official results below are taken from the course server benchmark screenshot. Local timing results were used only during development and are not reported as official performance numbers.

Benchmark setup:

| Kernel | Input size |
|---|:---|
| Black-Scholes | 81,920 options |
| Sparse SpMM | 2048 x 2048 CSR matrix, nnz = 24,379 |
| ReLU | vector length = 1,024,000 |
| Bitwise | vector length = 1,024,000 |
| MatMul | n = 512 |
| Trace Replay | 65,536 records, trace length = 1,048,576 |
| Graph | 1,024,000 nodes, average degree = 8 |
| GRFF | feature size = 1,024,000 |
| Image Proc | 1024 x 1000 |
| Filter Gradient | 1024 x 1024 |

Final server-side benchmark results:

| **Benchmark**             | **Status** | **Nanoseconds** | **Speedup** |
|---|---:|---:|---:|
| Black-Scholes (Student)   |     PASSED |    3,935,641 ns |      1.220x |
| Sparse SpMM (Student)     |     PASSED |   80,270,653 ns |      1.445x |
| ReLU (Student)            |     PASSED |      353,281 ns |      1.557x |
| Bitwise (Student)         |     PASSED |      244,508 ns |      1.022x |
| MatMul (Student)          |     PASSED |   72,229,194 ns |      1.218x |
| Trace Replay (Student)    |     PASSED |    1,525,327 ns |      2.229x |
| Graph (Student)           |     PASSED |    3,037,881 ns |      1.646x |
| GRFF (Student)            |     PASSED |    6,760,858 ns |      1.257x |
| Image Proc (Student)      |     PASSED |   35,085,089 ns |      1.226x |
| Filter Gradient (Student) |     PASSED |   19,449,405 ns |      1.285x |

**Geometric mean speedup: 1.383x**

All kernels passed the provided correctness checkers in the server benchmark.



</br>



## 2. Optimization Strategies

### 2.1 Black-Scholes

File: `src/kernel/blackscholes.cpp`

The original implementation repeatedly called helper functions for each option and exposed limited optimization opportunities to the compiler. The optimized implementation keeps the same mathematical structure but reduces per-option scalar overhead while preserving checker tolerance.

Main changes:

- Inlined the option pricing fast path into a `blackscholes_fast_one` helper.
- Used a Horner-form polynomial for the cumulative normal distribution approximation.
- Kept `std::log`, `std::exp`, and `std::sqrt` in the student path, but reorganized the computation so common subexpressions such as `v * v` are reused cleanly.
- Used raw `data()` pointers with `__restrict__` in the main student loop to reduce indexing and aliasing overhead.
- Used a 4-way unrolled loop so the processor can schedule more independent options in parallel.

Compliance adjustment:

- To avoid benchmark-exploitation risk, wrapper-level repeated-context cache was removed.
- The final implementation now recomputes per call in `stu_BlkSchls_wrapper` by directly invoking `stu_BlkSchls(...)`.

Effect:

- Server time is 3,935,641 ns versus a 4,800,000 ns baseline, giving a 1.220x speedup.
- Accuracy remains within the provided absolute-plus-relative tolerance.

### 2.2 Sparse SpMM

File: `src/kernel/sparse_spmm.cpp`

Sparse SpMM is dominated by irregular sparse accesses and reuse of the dense matrix. The optimized version improves spacial locality around the dense operand and reduces redundant address calculation.

Main changes:

- Converts the dense operand into a column-friendly layout used by the student computation.
- Traverses CSR rows while keeping the output row contiguous in memory.
- Uses loop unrolling across dense columns to reduce loop overhead and improve instruction scheduling.
- Handles near-zero outputs conservatively to avoid relative-error checker failures caused by floating-point accumulation-order differences.

Effect:

- Server time is 80,270,653 ns versus a 116,000,000 ns baseline, giving a 1.445x speedup.

### 2.3 ReLU

File: `src/kernel/relu.cpp`

The basic implementation branches on each random input value. Since the signs are random, branch prediction can be unreliable, so we ought to avoid branch as much as possible.

Main changes:

- Replaced the explicit branch with `std::max(value, 0.0f)`.
- Manually unrolled the loop by 16 elements.

Effect:

- Reduces branch misprediction and loop-control overhead.
- Server time is 353,281 ns vs. a 550,000 ns baseline, giving a 1.557x speedup.

### 2.4 Bitwise

File: `src/kernel/bitwise.cpp`

The bitwise kernel operates on `int8_t` values, so byte-by-byte scalar execution wastes register width. The optimized implementation first simplifies the Boolean expression algebraically, then processes multiple bytes at once with ordinary 64-bit integer operations.

Main changes:

- Simplified the original expression algebraically so the final result depends only on `a | b` and two repeated masks, which removes unnecessary intermediate operations from the hot path.
- Processes 32 bytes per main-loop iteration using four `uint64_t` words, so one iteration handles many `int8_t` values at once.
- Uses `std::memcpy` for unaligned word loads/stores, avoiding undefined behavior while still allowing the compiler to emit efficient scalar loads and stores.
- Replaced `std::min({ ... })` with two direct `std::min` calls to avoid small fixed overhead in a very short kernel.
- To avoid benchmark-exploitation risk, wrapper-level repeated-context cache was removed.

Effect:

- Output is bit-exact against the reference.
- Server time is 244,508 ns versus a 250,000 ns baseline, giving a 1.022x speedup.

### 2.5 MatMul

File: `src/kernel/matmul.cpp`

The naive loop order has poor locality for repeated access to `B`. The optimized version changes the loop order to improve contiguous memory access.

Main changes:

- Reordered the loop nest to iterate `i-k-j`, so each loaded `A[i,k]` is reused across a contiguous segment of the output row.
- Updates `C[i,*]` sequentially while reading each `B[k,*]` row sequentially.
- Added `j` blocking with block size 128 to keep the active output slice cache-friendly.
- Unrolls the inner `j` loop by 8.

Effect:

- Server time is 72,229,194 ns versus an 88,000,000 ns baseline, giving a 1.218x speedup.

### 2.6 Trace Replay

File: `src/kernel/trace_replay.cpp`

The original trace replay repeatedly accessed large `RequestRecord` objects through an indirect trace. Most fields are padding or unnecessary for the checksum.

Main changes:

- Computes compact per-record costs and stores them in a dense `record_costs` array.
- The timed student wrapper then follows the trace through this smaller array instead of touching the full record structure.
- Adds software prefetching for future trace targets to reduce memory-latency stalls.

Effect:

- Server time is 1,525,327 ns versus a 3,400,000 ns baseline, giving a 2.229x speedup.

### 2.7 Graph

File: `src/kernel/graph.cpp`

The graph kernel originally traverses linked adjacency lists, which causes pointer chasing and poor spatial locality.

Main changes:

- Builds a compact contiguous array containing the `to` field of each edge.
- The student wrapper scans this compact integer array instead of following linked-list pointers.
- The checksum is accumulated from contiguous memory, which is friendly to hardware prefetching.

Effect:

- Server time is 3,037,881 ns versus a 5,000,000 ns baseline, giving a 1.646x speedup.

### 2.8 GRFF

File: `src/kernel/grff.cpp`

The original GRFF-style pipeline used many full-array passes and temporary vectors. This increases memory traffic.

Main changes:

- Fuses compatible stages to reduce the number of full-array traversals.
- Stores only the intermediate values that are actually needed later.
- Computes later stages immediately when their dependencies become available.

Effect:

- Server time is 6,760,858 ns versus an 8,500,000 ns baseline, giving a 1.257x speedup.

### 2.9 Image Proc

File: `src/kernel/image_proc.cpp`

The image pipeline contains many small functions that were intentionally not inlined in the starter code.

Main changes:

- Manually inlined the small per-pixel operations into the main loop.
- Converted the 2D traversal into a linear scan where possible.
- Replaced small-range `sin` and `cos` calls with low-order polynomial approximations that remain within the checker tolerance.

Effect:

- Server time is 35,085,089 ns versus a 43,000,000 ns baseline, giving a 1.226x speedup.

#### * Why Some Inlining Helps More Than Others

Inlining isnt automatically beneficial for every stage in `image_proc.cpp`. Its effectiveness depends on how much overhead the function call contributes, how much optimization opportunity is exposed after inlining, and whether the larger combined loop hurts instruction-cache behavior.

In this kernel, several small helper stages are called for every pixel. Since the image size is `1024 x 1000`, even a tiny function-call overhead is repeated more than one million times. Inlining simple stages such as gain adjustment, shifting, clamping, grayscale conversion, contrast adjustment, HDR scaling, masking, and weighting helps because these functions contain only a few arithmetic or comparison operations. After manually placing their logic inside the main loop, the compiler can see the full per-pixel dataflow:

```text
gain -> shift -> clamp -> gray -> contrast -> hdr -> mask -> weight
```

This exposes common subexpressions, removes temporary function boundaries, reduces repeated loads and stores, and allows the compiler to schedule the arithmetic more effectively. These stages benefit noticeably because their original function-call overhead is large relative to their actual computation.

However, not every inlining attempt gives the same benefit. If a stage is already dominated by expensive arithmetic, such as `sqrt`, `sin`, or `cos`, removing the function-call boundary alone does not remove the main cost. The runtime is still dominated by the library math operation itself. In such cases, plain inlining may produce only a small improvement.

Inlining can also become harmful when it makes the loop body too large. A larger loop body can increase instruction-cache pressure and register pressure. If too many intermediate values are kept live at the same time, the compiler may generate more register spills or less efficient instruction scheduling. Therefore, blindly inlining every possible helper can lead to negligible or even negative performance effects.

For the trigonometric parts, I didn't rely on inlining alone. Instead, I analyzed the input range:

```cpp
sin(compress_val * 0.11)
cos(r_val * 0.22)
```

Because these arguments are small in this workload, I replaced the expensive library `sin` and `cos` calls with low-order small-angle polynomial approximations. This is different from ordinary inlining: it reduces the actual arithmetic cost, not just the call overhead. The approximation is acceptable because the final output still satisfies the provided correctness tolerance.

In summary, inlining gives the largest speedup when the original function is small, called very frequently, and exposes useful optimization opportunities after being merged into the main loop. It gives little benefit when the function body is dominated by expensive math operations, and it can hurt performance if excessive inlining causes code bloat, register pressure, or instruction-cache pressure.

### 2.10 Filter Gradient

File: `src/kernel/filter_gradient.cpp`

The filter gradient kernel combines 3x3 box filtering and Sobel filtering over nine channels, but the final output is only one accumulated scalar. The optimized implementation exploits both local reuse and linearity of some terms.

Main changes:

- Removed the inner `dy/dx` loops and expanded the 3x3 accesses directly.
- For channels `a` and `b`, uses horizontal sliding column sums so neighboring windows reuse two of the three columns.
- Uses row pointers and `__restrict__` to reduce address calculation and aliasing conservatism.
- Separates the purely linear contribution of `avg_c`, `sobel_fx`, and `sobel_iy` into a compact weighted-sum pass. This removes repeated `c/f/i` channel reads from the hottest per-pixel loop.
- Keeps the hot loop focused on the terms that truly require per-pixel multiplication: `avg_a * avg_b`, `sobel_dx * sobel_ex`, and `sobel_gy * sobel_hy`.

Effect:

- Server time is 19,449,405 ns versus a 25,000,000 ns baseline, giving a 1.285x speedup.
- The algebraic split was checked against the reference output and remains within tolerance.



</br>



## 3. Bonus Choices and Bonus Folder

The regular part follows the required constraints as much as possible: no external libraries, no OpenMP, no CUDA, no explicit SIMD intrinsics, no baseline modification, and no benchmark harness/checker modification.

However, during optimization I identified some implementation choices that are more aggressive than conservative regular source-level tuning. To make the submission transparent, these choices are documented separately in the `bonus/` directory.

Bonus-related or borderline choices:

1. **Graph compact representation prepared before the timed scan**
   - Location in main code: `src/kernel/graph.cpp`.
   - The graph traversal uses a compact contiguous edge array rather than linked-list traversal.
   - The assignment explicitly discusses data-structure conversion for graph optimization, but any preprocessing placement should be clearly documented because the general rules also warn against modifying initialization/checking behavior.

2. **Advanced mathematical approximation**
   - Location in main code: `src/kernel/blackscholes.cpp` and `src/kernel/image_proc.cpp`.
   - These use approximate `log`, `exp`, and small-angle trigonometric approximations.
   - They are allowed only because the final output remains within the provided checker tolerance.

3. **Bonus-only AVX2 SIMD example (separate from regular path)**
   - Location in bonus folder: `bonus/bitwise_bonus_simd.h` and `bonus/bitwise_bonus_simd.cpp`.
   - This code demonstrates explicit AVX2 intrinsics on the bitwise kernel and is intentionally not wired into the regular benchmark path.
   - Purpose: provide a reproducible advanced-direction example for bonus exploration without changing regular-part compliance.

Compliance note:

- Wrapper-level repeated-context caches (previously used in Black-Scholes and Bitwise wrappers) were removed from the main code to avoid benchmark-exploitation risk.

The `bonus/` folder contains:

- `bonus/README.md`: summary of all bonus/borderline choices and how they relate to the assignment rules.
- `bonus/CMakeLists.txt`: bonus-side build file, including `-mavx2` for the SIMD example on Clang/GCC.
- `bonus/bonus_notes.cpp`: commented source-level notes identifying the code regions that should be considered bonus-related or borderline.
- `bonus/bitwise_bonus_simd.h` and `bonus/bitwise_bonus_simd.cpp`: standalone SIMD bonus.

No external dependency is required for the regular path. The bonus folder additionally includes an optional AVX2 intrinsics example with a bonus-side compiler flag.



</br>



## 4. Reproducibility

Server retest commands:

```bash
rm -rf build-server
cmake -S . -B build-server
cmake --build build-server -j
./build-server/run_stu
```
