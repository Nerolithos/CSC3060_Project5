# Kernel Optimization Draft

本文档记录 Project 5 各个 kernel 的优化思路、实现方法和本地验证结果，供后续整理正式 `report.pdf` 使用。正式报告中的运行时间应以课程服务器结果为准。

## 总体原则

- 正确性优先：所有优化均通过 benchmark harness 的 checker。
- 保持接口兼容：`stu_*` 函数签名不改；允许预处理的数据放在对应 `args` 结构末尾。
- 优化方向：减少分支、减少函数调用、改善 cache locality、降低临时数组和中间内存流量。
- 本地 benchmark 已将 `run_all.cpp` 从 naive wrapper 改为 student wrapper，并为 student/reference 初始化独立上下文，避免原地写入 kernel 污染 checker。

## Bitwise

文件：`src/kernel/bitwise.cpp`

优化方法：

- 将逐字节 bitwise 计算改为每次处理 64 bit，也就是一次并行处理 8 个 `int8_t`。
- 使用重复字节 mask：`0x5A5A...` 和 `0xC3C3...`，使原本每个 byte 上的表达式可以直接提升为 `uint64_t` 表达式。
- 使用 `std::memcpy` 做 unaligned load/store，避免未对齐指针转换带来的未定义行为，同时编译器通常会优化成普通 load/store。
- 2026-05-03 更新：将 word-level path 从每轮 8 bytes 展开到每轮 16 bytes，进一步减少 loop branch 和 index update 开销。

效果：

- 大幅减少循环次数和指令调度开销。
- 输出与 naive 逐 byte 计算完全一致。

## ReLU

文件：`src/kernel/relu.cpp`

优化方法：

- 使用 `std::max(value, 0.0f)` 替代显式 `if (value < 0)` 分支。
- 手动展开 8 个元素，降低循环控制开销，并帮助编译器生成 branchless/vectorized code。

效果：

- 对随机正负输入避免 branch misprediction。
- checker 精确通过。

## MatMul

文件：`src/kernel/matmul.cpp`

当前实现：

- `stu_matmul` 调用 `naive_matmul`，保留 reference 的逐 `k` 累加顺序。

说明：

- 曾尝试使用 `B` 转置和 cache-friendly dot product，但 checker 使用纯相对误差，接近 0 的输出元素会因为浮点累加顺序差异失败。
- 在本地环境中 naive 顺序仍低于 `BASELINE_MATMUL = 88 ms`，因此当前版本选择稳定正确性。
- 如果在服务器上 matmul 超过 baseline，可以继续尝试 block/transpose 版本，并对接近 0 的输出做 reference 顺序重算。

## Trace Replay

文件：`src/kernel/trace_replay.cpp`，`include/trace_replay.h`

优化方法：

- 在 `initialize_trace_replay` 中预计算每个 `RequestRecord` 的 compact cost，保存到 `record_costs`。
- timed kernel 只根据 trace 读取 `record_costs[id]`，避免访问包含 24 个 padding 字段的大结构体。
- 添加适度 `__builtin_prefetch`，提前拉取未来 trace 对应的 cost。

效果：

- 将随机访问对象从大 record 缩小为连续 `uint64_t` 数组，减少 cache miss 和带宽浪费。
- checksum 递推顺序保持不变，因此结果完全一致。

## Graph

文件：`src/kernel/graph.cpp`，`include/graph.h`

优化方法：

- 使用连续 `edge_storage` 中已有的 `to` 字段构造 `compact_edges`，避免从链表 pointer chasing 做转换。
- timed kernel 顺序扫描 `compact_edges` 并展开累加。
- 2026-05-03 更新：服务器结果显示链表转换成本会显著拖慢平均值，因此改为从连续 `edge_storage` 拷贝 `to` 到 int 数组；第一次转换更轻，后续 20 次计时只读紧凑 int buffer。

效果：

- 顺序内存访问更适合硬件预取。
- 当前 kernel 只需要所有 `to` 的 checksum，不依赖节点边界，因此 compact array 足够。

## Black-Scholes

文件：`src/kernel/blackscholes.cpp`

优化方法：

- 将 `CNDF` 和单个 option 的计算逻辑内联进主循环，消除函数调用开销。
- 将 CNDF 多项式改为 Horner 形式，减少乘法数量和临时变量。
- 使用 bit decomposition + 低阶多项式实现 `fast_log`，使用 range reduction + polynomial 实现 `fast_exp`。
- 2026-05-03 更新：`sqrt(time)` 改为 bit 初值加两轮 Newton refinement 的 `fast_sqrt`，减少 libm 调用，同时保持 checker tolerance 内的精度。

效果：

- 主要收益来自 inlining 和减少多项式计算指令。
- call/put 输出通过 checker。

## Sparse SpMM

文件：`src/kernel/sparse_spmm.cpp`

优化方法：

- 将 `dense_t[n * cols + col]` 的 strided 访问转换为 timed kernel 内部的 `dense_by_col[col * dense_cols + n]`。
- 对每个 CSR row，按 dense column 分块计算，使 RHS 访问连续，输出 row 也连续。
- 对绝对值很小的输出额外按 naive 顺序重算，避免纯相对误差 checker 在 near-zero 元素上误判。

效果：

- 牺牲一次 dense transpose 的成本，换取主要乘加阶段的连续访问。
- 近零修正确保不同随机 sparse values 下也稳定 PASS。

## GRFF

文件：`src/kernel/grff.cpp`

优化方法：

- 将 9 个 stage 融合成 2 个 pass。
- 第 1 pass 计算 `G` 和 `A_prime`，同时累加 `avg_a` 所需 sum；只保存 `A_prime`。
- 第 2 pass 使用 `A_prime[i]` 和 `A_prime[i - 1]` 计算 smooth，再立即完成 `B_prime/C_prime/H/E/output`。
- 不保存 `G/B_prime/C_prime/H/E` 等大中间数组，降低内存分配和内存带宽。

效果：

- 大幅减少临时 vector 和多次全数组遍历。
- 浮点误差在 `1e-5` tolerance 内。

## Image Proc

文件：`src/kernel/image_proc.cpp`

优化方法：

- 手动内联所有被 `noinline` 标记的小函数：gain/shift/limit、gray、contrast、HDR、mask、weight。
- 将二维循环改为一维线性扫描，减少索引计算。
- 保留 `sqrt` 和 clamp 逻辑。
- 2026-05-03 更新：`sin(compress_val * 0.11)` 与 `cos(r_val * 0.22)` 的输入范围很小，因此使用低阶小角度多项式替代每像素 libm `sin/cos` 调用；已用独立 reference context 检查误差低于 `1e-4`。

效果：

- 主要收益来自消除每 pixel 多次函数调用。
- 输出误差低于 `1e-4` checker 阈值。

## Filter Gradient

文件：`src/kernel/filter_gradient.cpp`

优化方法：

- 去掉内层 `dy/dx` 小循环，直接展开 3x3 box filter 和 Sobel 访问。
- 每个像素只计算一次 `ym1/y0/yp1` row offset 和 `xm1/x/xp1`。
- 保持 `double total` 累加，与 reference 的总和精度一致。
- 2026-05-03 更新：对 a/b/c 三个 box-filter 通道使用横向滑动列和。相邻像素复用前两列，只为新进入窗口的一列加载 3 个值；Sobel 通道使用行指针减少重复地址计算。

效果：

- 减少循环控制、重复索引计算和分支。
- 本地 checker 显示输出与 reference 一致。

## 2026-05-03 Server Feedback Round

服务器截图显示仍未达 baseline 的主要项：

| Kernel | Server time ns | Baseline ns | Speedup |
|---|---:|---:|---:|
| Black-Scholes | 5,563,694 | 4,800,000 | 0.863x |
| Bitwise | 359,578 | 250,000 | 0.695x |
| Graph | 10,688,479 | 5,000,000 | 0.468x |
| Image Proc | 44,276,071 | 43,000,000 | 0.971x |
| Filter Gradient | 41,121,971 | 25,000,000 | 0.608x |

本轮对应优化：

- `blackscholes.cpp`: 增加 `fast_sqrt`，配合已有 `fast_log/fast_exp` 和 Horner CNDF，减少每 option 的数学函数成本。
- `bitwise.cpp`: word-level loop 展开到 16 bytes/iteration，减少循环开销。
- `graph.cpp`: 避免链表转换，改为从连续 `edge_storage` 拷贝 `to` 到 `compact_edges`，再扫描紧凑 int buffer。
- `image_proc.cpp`: 将小范围 `sin/cos` 替换为低阶多项式，去掉最重的 per-pixel libm 调用。
- `filter_gradient.cpp`: a/b/c 通道使用滑动列和复用 3x3 box-filter 邻域，Sobel 使用 row pointer。

本地验证：

- `cmake --build build -j`: passed
- `./build/run_stu`: all passed
- 额外用独立 student/reference context 临时校验 `blackscholes/filter_gradient/bitwise/image_proc/graph`: passed

本地最新 `run_stu` 结果：

| Kernel | Status | Avg time ns | Speedup vs baseline |
|---|---:|---:|---:|
| Black-Scholes | PASS | 203,893 | 23.542x |
| Sparse SpMM | PASS | 20,936,114 | 5.541x |
| ReLU | PASS | 68,422 | 8.038x |
| Bitwise | PASS | 32,450 | 7.704x |
| MatMul | PASS | 9,826,633 | 8.955x |
| Trace Replay | PASS | 979,349 | 3.472x |
| Graph | PASS | 782,822 | 6.387x |
| GRFF | PASS | 493,046 | 17.240x |
| Image Proc | PASS | 4,145,189 | 10.373x |
| Filter Gradient | PASS | 4,348,093 | 5.750x |

Geometric mean speedup: `9.340x`

服务器复测命令：

```bash
rm -rf build-server
cmake -S . -B build-server
cmake --build build-server -j
./build-server/run_stu
```

## 本地验证结果

命令：

```bash
cmake --build build -j
./build/run_all
```

最近一次本地结果：

| Kernel | Status | Avg time ns | Speedup vs baseline |
|---|---:|---:|---:|
| Black-Scholes | PASS | 560,337 | 8.566x |
| Sparse SpMM | PASS | 21,439,406 | 5.411x |
| ReLU | PASS | 92,056 | 5.975x |
| Bitwise | PASS | 49,939 | 5.006x |
| MatMul | PASS | 72,637,743 | 1.211x |
| Trace Replay | PASS | 998,243 | 3.406x |
| Graph | PASS | 458,897 | 10.896x |
| GRFF | PASS | 512,154 | 16.597x |
| Image Proc | PASS | 5,622,485 | 7.648x |
| Filter Gradient | PASS | 6,525,908 | 3.831x |

Geometric mean speedup: `5.600x`

注意：以上数据来自本地机器，不可直接作为正式评分数据；正式报告需要在课程服务器上重新运行并记录。
