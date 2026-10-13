# PTO Flash Attention：四阶段 CV 流水、在线 Softmax 与 FIFO 预执行

这篇研究官方 PTO 正式 **v9.0.0** 的 `kernels/manual/common/flash_atten`。commit 为 `377526952e95c39f9149cb94d261eba9e394f0d4`，正式配套表将 cann-pto-isa v9.0.0 对应 **CANN 9.0.0**；案例文档明确支持 Ascend 910B / 910C，也就是 A2/A3。A5 有另一个目录，本篇不使用它。[固定原始案例](https://gitcode.com/cann/pto-isa/blob/377526952e95c39f9149cb94d261eba9e394f0d4/kernels/manual/common/flash_atten/README_zh.md)。

## 1. 四阶段实现，不把 S0×S1 完整分数写入 GM

`fa_performance_kernel.cpp` 将一次 Attention 组织成：

```text
compute_qk (Cube) → compute_p (Vector Softmax)
                 → compute_pv (Cube) → compute_gu (Vector running update)
```

Q/K/V 输入为 FP16，QK/PV 累加采用 FP32。`compute_qk` 按 S1 tile 处理当前 Q 行块与 K；Q 的矩阵 tile 在第一轮加载后驻留，后续主要推进 K。`compute_p` 产生当前 tile 的指数值，转为 FP16 供第二次矩阵乘；`compute_pv` 计算部分输出，`compute_gu` 累加并在最后一个 tile 归一化。

使用 FIFO 和固定 tile 减少完整分数矩阵的保存，并不表示 A2/A3 的 Cube/Vector 之间完全没有 GM staging。固定代码的 FIFO 路径含全局缓冲；不能把 950 片上交换能力写成本例的优化效果。

## 2. 在线 Softmax 的数值状态也进入流水设计

Vector Softmax 保存每行历史最大值 M 和指数和 L。处理新分数 tile 时，先取当前行最大值 m，更新 `M_new=max(M,m)`，计算历史重标定系数 `exp(scale*(M-M_new))`，再得到当前指数 `exp(scale*(score-M_new))`。历史 L 和 running output 都必须乘重标定系数，才能与当前 tile 累加。

`pto_macro_fa_softmax.hpp` 和 `pto_macro_fa_gu.hpp` 将这些步骤展开为 `TROWMAX / TROWSUM / TROWEXPANDSUB / TROWEXPANDMUL / TROWEXPANDDIV` 等操作。FIFO 不仅传 P 和 PV，还要让正确的 exp-max 状态与对应 tile 配对。如果只给 QK/PV 加深队列，却没有同步调整状态环形索引，算子可能跑完但得到错误输出。

## 3. qkPreloadNum：让 Cube 领先，给 Vector 腾出重叠空间

Kernel 先预执行若干 QK/Softmax tile，再进入稳态循环，在后续 QK 与先前 PV/GU 间交错。这比每个 tile 必须完整走完四阶段之后才开启下个 tile 更有机会使用 Cube 与 Vector 的并行执行。

固定源码的参数有明确边界：`qkPreloadNum>=1`；`TILE_S1` 必须能被 `CUBE_S1` 整除；`Cube_S0` 必须能被 Vector 子核数与 tile 切片因子整除。`srcVecTNBuffers / xexpVecTNBuffers / outOTileNBuffers` 和 K/P/V 的矩阵缓冲使用 ping/pong。FIFO 深度在当前代码来自 **CV_FIFO_SIZE**，并非无条件等于 `1+qkPreloadNum`；旧 README 的简化关系不能替代固定 Kernel 定义。

`TPUSH / TPOP` 处理生产与消费依赖；`pending_consumption_events` 计算还未 drain 的事件，在末轮补齐。扩展 preload 的收益与状态、FIFO、L1/UB/GM footprint 增加同时发生。调参应该先固定 shape，比较合法的 preload=1/2/4，再查看是否减少 stall，而不是默认越深越快。

## 4. 多核切分与每核 tile 必须分开

`block_rows=S0/CUBE_S0` 决定 Query 行块的并行数；`num_tiles_s1=S1/TILE_S1` 决定每个行块的归约循环。每个逻辑 S1 tile 还可拆成若干 `CUBE_S1` 子 tile。改变 S1 或 TILE_S1 后，不能继续引用旧图中的“循环次数”。

多核主要沿 Query 行分工，不等价于沿 S1 做 split reduction。较小 S0 下没有足够 Query 块时，更多物理核不会自动提高吞吐。Causal mask 会进一步改变各 Query 块的工作量，默认非 causal 的数据不能迁移成 causal 延迟。

## 5. 正式 tag 的性能表：绝对时间与归一化要分清

该 tag 的文档提供 A2/A3 的以下参考时间，S0 随使用核数变化：

| 使用核数 / S0 | S1=1024 | S1=2048 | S1=4096 | S1=8192 |
|---|---:|---:|---:|---:|
| 1 / 128 | 42.08 μs | 51.44 μs | 43.80 μs | 74.54 μs |
| 2 / 256 | 33.201 μs | 44.101 μs | 43.521 μs | 75.162 μs |
| 4 / 512 | 41.721 μs | 55.441 μs | 46.621 μs | 86.322 μs |
| 8 / 1024 | 63.72 μs | 85.882 μs | 64.461 μs | 107.342 μs |

这不是同一工作量的强扩展实验，因为 S0 同时改变。文档 `Normalized TFLOPS=measured TFLOPS×24/cores_used` 是线性归一化，不是实际测得的满 24 核吞吐。表格只标 A2/A3 范围，没有逐项标明具体板卡 SKU；测量 CANN、驱动、当时 commit 也未单列，不能把正式 tag 兼容性当成这些历史数字的精确测量版本证明。

还有一个原始单位问题：Host 的 `gops=4*S0*S1*HEAD_SIZE/1e6` 实际得到 **MOp**，虽然 CSV/README 标签写 GOps。它除以微秒得到的数值可对应 TFLOPS，但把该变量当 GOp 再自行换算会差 1000 倍。本篇保留原文件并指出此问题，不重新包装原表为不存在的性能结论。

## 6. Host 计时是设备跨度，不是 shell time

Kernel 记录各 Cube/Vector block 的开始和结束系统计数器。Host 取最早 start 与最晚 end 的跨度，默认 A2/A3 用每 tick 20 ns 转换，再另算各 block 的平均 duration。这个 span 不等于平均 block 时间，更不包含构建和生成数据。改变设备/仿真环境时还必须确认 tick 换算，不能拿错误的换算系数讨论加速。

提供的 wrapper 固定小 case `HEAD=128,S0=128,S1=1024,CUBE_S0=128,TILE_S1=128`，可分别运行 preload=1 和 4。先检查最终 O golden；`--intermediate` 可检查中间状态但不用于正式性能成绩。再记录设备 span、同规格多次样本和完整版本环境，使用 msprof 确认实际重叠。

本机没有 CANN/NPU，源码、原始流水图、数据生成、Host 计时逻辑和 wrapper 静态检查已保存；未宣称复现上述数字。作为完整注意力内核，这个案例的价值是把数值递推、生产消费顺序、缓冲容量与多核分工放在同一份可检查实现里。[复现说明](reproduce.md)。
