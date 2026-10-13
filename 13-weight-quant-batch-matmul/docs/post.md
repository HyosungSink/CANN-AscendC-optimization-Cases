# A2/A3 权重量化 GEMM：NZ Split-K 的四槽反量化环形流水

本篇拆解 `WeightQuantBatchMatmulV2CustomNzSplitkKernel`：浮点激活配合低比特权重，在 Vector 侧分块反量化权重，再由 Cube 计算。重点是 **NZ 权重、按 K 分核、四槽 GM 环形缓冲和 A 全载**，不是整网量化方案。

上游固定 ops-nn 正式 `v9.0.0`，提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。官方 [CANN 9.0.0 配套表](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) 明确支持该标签；Host def 注册 `ascend910b`、`ascend910_93`，Kernel 入口在 A2/A3 编译架构分支中包含本文实现。未找到固定版本的公开 A/B 耗时，本机无 NPU。

## 1. 为什么权重反量化也要流水

权重 INT8/INT4 可以降低 HBM 容量和搬运量，但 Cube 需要浮点计算时，权重必须先恢复成浮点。若先生成整份浮点权重，内存成本又变大；若每个 Cube 都停下来等待 Vector 恢复当前块，流水容易串行。

该实现用受限大小的环形 workspace 连接两侧：Vector 为未来的 K/N 片段生成反量化权重，Cube 消费已就绪的片段，并在消费后回传槽位可复用的信号。权重仍然经过 GM→UB→GM→L1，不能把它称为“零 HBM 中间结果”；价值在于分块和生产消费重叠。

## 2. 这不是所有 W8/W4 shape 都能用的模板

[Host CustomNzSplitK](../code/upstream/matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.cpp) 的 `IsCapable` 明确限制：

- **65 ≤ M ≤ 256**，N/K 不超过 65535。
- A 不转置、B 转置；权重必须是真实 FRACTAL_NZ。
- 只支持 per-channel 反量化；不走输出 INT8 quant，也不接受 UINT64/INT64 的反量化 scale 类型。
- N/K 通常按 64 对齐，另外保留少数网络 shape 白名单例外。
- 请求确定性计算时拒绝此模板。

这组约束说明算子级优化不能只写“用了 NZ/Split-K”：需要同时记录模板命中条件、数据布局和数值约束。通用算子也有其他 ND、MSD、Fixpipe 分支，本文不混用它们的能力或性能。

## 3. K/N 怎样分给核

`GetMatMulTiling` 先按 `DEFAULT_SINGLE_CORE_SIZE=1024` 估算 K 分核，再根据总 AIC 数确定 N 分核。Cube 基本块 M/N/K 通常为 128/128/128，Vector 的单 N 段为 32，K 的生产消费步长 `singleK=512`。

两个 Vector 子核各处理当前 Cube N 块的一半，避免两者反量化同一片权重。大 K 通过多个 Cube K 分片增加并行度；各核输出的部分和需要最终合并。它适合输出 tile 不够多、K 较长的情况，但小 K 上额外的中间缓存和同步可能得不偿失。

Host 还检查 `M*singleCoreK*sizeof(A) ≤ 256KiB`。满足时开启 A 的 L1 全载模式，让同一 K 分片的激活能在后续 N 块间复用；这是一半 L1预算的具体判断，不能当成全部激活在全设备常驻。

## 4. 四个槽位怎样避免读写冲突

[实际 Kernel](../code/upstream/matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom_nz_splitk.h) 设 `WEIGHT_CACHE_COUNT=4`。`ProcessVector` 使用 `totalLoopIdx % 4` 选择 GM 中的权重槽位，`AntiquantWeight` 写当前反量化块。

第一到第四块可以推进到不同槽位；当 Vector 要绕回覆盖槽位时，它通过 `CrossCoreWaitFlag(VEC_CUBE_FLAG_ID)` 等待 Cube 已消费。生产完成后以 `PIPE_MTE3` 通知 `CUBE_VEC_FLAG_ID`。

Cube 的 `CopyInBL1` 先等生产通知，从对应槽位把浮点权重载入 L1。MTE2 消费完成后，若后续还会复用槽位，设置 `VEC_CUBE_FLAG_ID` 允许 Vector 覆盖。重要的是“读完成就可以归还 GM 槽位”，而非必须等待整次矩阵乘结束；L1里已有自己的副本。

权重 UB输入和 scale/offset使用 ping-pong缓冲，相关 `V_MTE2、MTE2_V、M_MTE1、FIX_M` HardEvent保护缓冲生命周期。四槽 workspace 与两份 UB/L1局部缓冲是不同结构，不能把它们一律说成 Double Buffer。

## 5. Split-K 的后处理与确定性取舍

当 K 跨核、输出为 BF16 或存在 bias 时，`hasPostProcessFlag_`要求后处理。初始化函数先清零累加workspace，Cube通过 FP32 AtomicAdd写入部分和，待相关核完成，再由 Vector `PostProcess`转换输出/处理bias。

浮点原子累加的到达顺序可能变化，因此 Host 明确不支持确定性要求。这里的速度取舍不仅是内存与核数，还包括结果一致性。需要确定性时应选上游另一条受支持路径，而不是删除 `IsCapable` 的检查。

## 6. 应该如何验证

先用 M=128、N/K为64倍数、A不转置/B转置、NZ低比特权重和per-channel scale构造输入，并核对tiling确实选中本文模板。M=64/65/256/257是检查模板边界的关键shape；N/K白名单例外应单独记录。官方默认示例未必走NZ Split-K，只能作为API smoke test。

比对同样量化权重离线恢复的浮点参考，记录误差、bias与offset约定。计时要排除NZ打包和初始化；用profiler观察Vector生产、GM环形槽位等待、Cube读取与AtomicAdd后处理，不能只报告总wall-clock。

[来源](sources.md)、[原始源码](../code/upstream/) 和 [复现](reproduce.md) 都固定提交与依赖。当前只有实现证据，没有已验证的加速倍数。
