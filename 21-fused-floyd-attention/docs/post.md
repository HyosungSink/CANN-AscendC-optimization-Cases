# Floyd Attention 的 CV 分块与在线累积

Floyd Attention 的打分来自两条 Query/Key 配对，输出也来自两条概率/Value 配对。将它按小算子逐步拼接，会物化较大的打分和概率张量。本案例研究 CANN 9.0.0 的 `FusedFloydAttention` 如何把两路矩阵乘、Softmax 与输出累积组织成单个融合算子，并把 Cube 和 Vector 的基本块分别设计。

源码固定正式 `v9.0.0` 标签，commit `afe72144f9f2ac8441929035795db88a111b30c5`。产品表支持 A2 训练系列和 A3 训练/推理系列；A2 推理系列被明确标为不支持。Host 实际注册 `ascend910b` 和 `ascend910_93`，复现须同时遵守产品与 shape 约束。[正式版本映射](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)、[算子说明](../code/upstream/attention/fused_floyd_attention/README.md)

## 把两条打分支路合到同一个 Softmax

数学上先计算 `scale × (QK1 + QK2)`，应用 mask，再得到一份概率 P；输出为 `PV1 + PV2`。两路分数必须先相加再做 Softmax，不能对各支路独立归一化之后合并，否则语义不同。

前向实现包含两路 QK 与两路 PV。发布说明提到的十个 Matmul 是对 Floyd 正反向融合的整体说明，不能将它当作这里前向每次循环的固定 Matmul 数量。主文件 `fused_floyd_attention_s1s2_bn2gs1.h` 提供实际执行顺序。

## Cube 和 Vector 不必使用相同块大小

上游设计说明使用 Cube `[32,1024]`、Vector `[8,1024]` 的示例来解释 CV tiling。一个 Cube 结果由 Vector 分成四段处理，Vector 的一个 FP32 基本区域为 `8×1024×4=32 KiB`。Cube 可以连续处理较大矩阵乘，Vector 则以适合 UB 预算的小块完成缩放、mask 和 Softmax。

这些尺寸是设计示例；实际 Host 根据维度、nRatio 和 UB/L1 限制计算参数，不能把 `[32,1024]` 固定到所有 shape。此处扩大的是 Cube 一次交付的工作量，不是取消 GM 交互：源码的 `mm1Res`、`mm2Res` 和 `vec2Res` 等 workspace 仍然存在。[设计说明](../code/upstream/attention/fused_floyd_attention/docs/FFA算子设计介绍.md)、[Host Tiling](../code/upstream/attention/fused_floyd_attention/op_host/fused_floyd_attention_tiling_general.cpp)

## 三个任务记录把生产和消费错开

`Process` 使用 `SplitExtraInfo extraInfo[3]`。当前 task 推进两路 QK；上一 task 处理 Vec1，再启动两路 PV；上两轮 task 等待 PV 完成并处理 Vec2。`taskId%3` 对应三个任务的参数记录，而结果 workspace 还按 `taskId%2` 使用两个槽。这两种环形结构保护不同对象。

主循环末尾额外推进两轮，用来排空仍在流水中的 Vec1/PV/Vec2。`MTE3_MTE2` 等事件保障 Vector 写出的概率数据能够被后续矩阵乘读取。减少同步或删去排空迭代，必须先证明这些消费者仍然能读到完整数据。[主流水源码](../code/upstream/attention/fused_floyd_attention/op_kernel/fused_floyd_attention_s1s2_bn2gs1.h)

```mermaid
flowchart LR
    A[两路 QK 得分相加] --> B[缩放与 mask]
    B --> C[在线 Softmax]
    C --> D[两路 PV]
    D --> E[修正并累计输出]
    E --> F[最终除以 Softmax sum]
```

## 在线 Softmax 限制中间张量大小

KV 轴分块后，每块维护 max 和 sum。新块的最大值改变时，旧的未归一化输出乘上 `exp(oldMax-newMax)`，再加本块 PV。最终一次按行除以总 sum，避免每块都先做完整归一化并保存整张概率矩阵。

这一模式减少大张量物化，又保持跨块数值稳定。`softmax_max` 和 `softmax_sum` 作为输出保留给反向使用，因此算子并没有消除所有统计张量。全 mask 行、最后一块、不同 D 的 Cast 与累计误差应分别验证，不能只检查最终张量 shape。

## Batch 在中间轴时的搬运取舍

Floyd 的某一类 BatchMatmul 的 batch 轴在中间，单个 batch 的数据不一定连续。实现使用 `IterateBatch` 的布局与 stride 参数，区分 NORMAL 和适用的 BSNGD 表达，避免仅靠一个普通连续指针描述整个输入。

上游附录给出容量约束：`oneBatchBytes = 元素字节 × D × (K1+K2)`，可同时装入 L1 的 batch 数为 `floor(L1Size/oneBatchBytes)`。该说明以 910B 的 512 KiB L1 为例。实际代码读取平台容量；改变 batch chunk 时必须保持 L1 预算与矩阵步幅一致。减少调用次数的收益要与更大工作集的容量和流水成本一起评估。

## 上游公布了具体的算子级对照

固定标签中的设计文档比较 Chunk 实现与 Ascend C 融合实现，原始时间单位为秒，以下换算为毫秒：

| B H N M K D | Chunk | Ascend C | 原文加速比 |
| --- | ---: | ---: | ---: |
| 1 32 256 256 256 32 | 28.2 ms | 20.9 ms | 1.35× |
| 1 32 768 768 768 32 | 764.0 ms | 456.3 ms | 1.67× |
| 1 16 384 384 384 64 | 56.1 ms | 31.6 ms | 1.78× |

同一文档给出 `[1,32,512,512,512,32]` 的显存占用由 45.50 GB 到 9.50 GB，原文节省率 79.12%。这是原表的显存口径，不能直接解释为算子的 workspace 减少 79.12%。来源未完整列出测量时的 CANN、驱动、baseline 源码及峰值内存采集方法；保留这些数字作为上游工作负载参考，不将其改写为本机或任意 9.0.0 环境的实测保证。

## 复现应该同时固定 baseline 的切块策略

先构建正式快照的融合算子，跑 aclnn 样例和 Host UT。数值参照应完整保留两路得分、mask 和两路 Value 的语义。再用相同 dtype、输入和 scale 比较完整的拆分链与融合 kernel，记录拆分链的 Chunk 大小、是否物化全量打分以及运行期内存峰值。

建议改变 D=32/64/128 和序列维度，观察 nRatio、CV 子块数、Matmul 等待、MTE 搬运与 Vec1/Vec2 的关系。文章中的公开速度差异是整个融合实现相对 Chunk 实现的结果，不是单独启用 Double Buffer 的消融结论。构建及验收入口见 [复现说明](reproduce.md)，原始表格和源码出处见 [来源记录](sources.md)。
