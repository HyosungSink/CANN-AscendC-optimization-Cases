# 当前 Kernel 调度说明

本文对应 [kernel.asc](../code/kernel.asc) 当前版本，不再描述早期的手写 C220 单一路径。

## 入口调度

`run_kernel` 先读取 `M/N/K`，决定是否补齐 K、选择小矩阵专用路径和 Cube tiling。

```text
if M<=128 and K in {32,64,128} and N in {16,32,64,128}:
    SmallKernel family
else if K>65535:
    HugeKKernel (AIV correctness fallback)
else:
    MatmulApiTiling -> Cube/AIV path
```

通用路径的关键参数是：

- `tileM`：按可用 Cube 核数从 96 限制范围内计算；通用路径的最小值为 8，避免非对齐
  M 形成不稳定的行槽布局。
- `baseN`：通常为 128；K 补齐的小 K 路径使用 256，特殊大 K 场景按条件使用 256。
- `tileN`：宽矩阵按可用核数分区，减少一个短行任务占用全部列的情况。
- `CFG_MDL/CFG_NORM`：大 N 或大 K 使用 MDL 配置，其余使用普通配置。

## Cube/AIV 路径

`FusedKernel` 让 AIC 通过 `CubePipeline` 处理一组完整的 N，AIV 以两个 GM
accumulator slot 交替处理行块。`WideKernel` 将 `(row tile, column tile)` 作为独立
任务，适合行块少于可用 Cube 核的短而宽矩阵。

当 `N` 在 1024 到 8192 且 `rowTiles` 不超过可用核数、K 较大时，选择
`ColumnStreamKernel`；它按 512 列组处理最后的非完整列组。当 `N` 约为 16K、K 较小且 tile 条件满足时，选择
`LongColumnStreamKernel`。这些路径按列组通知 AIV，AIV 先生成浮点 activation，再
统一计算每行 yScale 和量化结果。

## 量化和精度

AIV 保持以下顺序：INT32 -> FP32 -> 乘 `x1Scale` -> 乘 `x2Scale` -> ReLU -> 行最大值。
量化阶段使用 FP32 除法和 round-to-even。硬件向量除法在半整数附近可能产生边界误差，
源码通过位级候选检测和 `RoundedNearHalfBits` 对候选元素重新计算，避免只依赖
`CAST_RINT` 导致 bit mismatch。

`Small*` 和 `HugeKKernel` 也保持相同的行最大值、scale 和舍入语义；只有计算载体不同。

## 边界与 workspace

- `PadInputsKernel` 处理 `N>=16384`、小 K 非 32 对齐的输入，补零后再运行 Cube；
  8192 到 16383 列不再为一次额外 Pad 启动付固定开销。
- Cube 输出写到 `aclrtMalloc` 的 INT32 accumulator；AIV 读取并写出最终 y。
- padded A/B 放在 accumulator 分配尾部，大小由 `m+n` 和补齐后的 K 计算。
- `aclrtSynchronizeStream` 后释放这块分配，不能在 kernel 异步未完成时提前释放。
- `K>65535` 不进入 Matmul API，避免 tiling 字段截断；该分支覆盖 `65536`、`65537`
  以及更大的 K。

所有路径都必须通过非对齐 N、尾 M、尾 K、半整数 tie 和大 K 对拍后才能保留。
