# 当前实现架构

当前版本采用“host 侧形状调度 + Cube/AIV 专用 kernel”的架构。它不是旧版的单一
手写 Mmad kernel，也不是无 workspace 方案；设计重点是让常见大矩阵使用 CANN
Matmul API，同时为小矩阵、宽矩阵和超大 K 保留可验证的专用路径。

## 数据流

```text
x1/x2 (INT8, GM)
        |
        +--> PadInputsKernel (仅小 K 非对齐且 N>=16384)
        |
        +--> Cube Matmul -> INT32 accumulator (GM)
        |                         |
        |                         +--> AIV epilogue
        |                              FP32 dequant -> ReLU -> row max
        |                              -> yScale -> round-to-even INT8 y
        |
        +--> Small*/HugeKKernel (AIV-only special cases)
```

`x2` 以 `(N,K)` 存储，因此逻辑矩阵乘为 `x1 @ x2.T`。所有路径都输出同一语义的
`y` 和 `yScale`。

## 路径选择

| 条件 | Kernel | 目的 |
|---|---|---|
| 小 M、`K<=16` 或 `K=32/64/128` 的短/非对齐面板 | `SmallKernel` family | 物理面板补齐，复用 B，绕过 Cube 固定开销 |
| `K>65535` | `HugeKKernel` | 避免 Matmul tiling 16 位边界，优先正确性 |
| 短行、宽 N | `WideKernel` | 将 row/column tile 分发到更多 Cube 核 |
| `N=1024..8192`、`K>=512` 且非宽行 | `ColumnStreamKernel` | AIC 按 512 列组生产，AIV 流式消费，支持尾组 |
| 约 16K 列、小 K | `LongColumnStreamKernel` | 4096 列组流式处理 |
| 其他常见尺寸 | `FusedKernel` | Cube 行块与 AIV epilogue 协同 |

## 内存和同步

host 侧为 INT32 accumulator 分配足够空间；`workRows` 根据宽矩阵和普通矩阵分别计算。
若需要 K 补齐，padded A/B 紧接着放在同一分配尾部。设备侧使用 cross-core flags
让 AIC 的 Fixpipe 写入和 AIV 的读取按 slot 交替进行，避免覆盖仍在使用的行块。

UB 中的 epilogue buffer 按 N 和批量行数动态配置。列流式路径先把反量化后的 FP32
分块写回 accumulator 的 float 视图，再由 AIV 完成行归约和量化。所有 GM 写入均按
真实 M/N 尺寸进行，尾块通过 `DataCopyPad` 处理。

## 已知边界

1. accumulator/padded input 使用的是本轮方案明确申请的设备 workspace；不能声称为
   “无 workspace”。
2. `K>65535` 走标量 AIV 兜底，精度通过但吞吐较低。
3. `msprof` 在当前环境可以得到 kernel 时间，但无法稳定导出 summary，因此文档只
   记录可复现的 kernel 时间和端到端测试结果，不虚构硬件利用率。

## 验证标准

每次修改必须运行：

```bash
python3 ../scripts/test_dimensions.py
python3 ../scripts/test_dimensions.py --patterns
python3 ../scripts/test_dimensions.py --boundaries
```

当前结果为 15/15、12/12、22/22 全部 PASS，主测试估算均分约 20.592（仅作本地版本
比较，不等同线上隐藏形状得分）。详细逐点耗时、
硬件信息、编译命令和失败实验见 [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)。
