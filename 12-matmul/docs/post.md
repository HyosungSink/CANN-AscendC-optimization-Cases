# 小 M、大 K 的 FP32 Matmul：把 A 全载到 L1，而不是反复读取

本篇研究 ops-nn **CANN 9.0.0 正式版** `matmul/mat_mul_v3` 中一个非常具体的分支：A2/A3 上 `DoAL1FullLoadTiling → MatmulBaseKernelAL1FullLoad`。不是“优化 GEMM”的泛泛建议，而是明确到 dtype、转置、shape 和片上容量约束的实际实现。

源码固定 `v9.0.0` 提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`；硬件注册包含 `ascend910b`、`ascend910_93`。官方 [9.0.0 配套说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) 将 ops-nn 9.0.0 对应到该正式标签。未取得公开 A/B 单算子耗时，也未在本机运行 NPU；本篇提供真实源码机制和可验证实验入口。

## 1. 面对的瓶颈

若 M 很小、K 很大，而 N 被切给多个核，A 矩阵会被重复搬入。与其为每个 N 基本块重新搬 A，适合的 shape 可以让每个工作的 Cube 核先载入完整的本核 A，在后续列块之间复用。

注意复用是**每个核自己的 L1**。各核之间没有共享 L1，不能声称整个 A 在全设备只从 HBM 读取一次。这个分支也不是任意 BF16 或 FP16 GEMM 的普遍优化，它针对 FP32 小 M 场景。

## 2. Host 用什么条件选择

阅读 [matmul_v3_base_tiling.cpp](../code/upstream/matmul/mat_mul_v3/op_host/op_tiling/matmul_v3_base_tiling.cpp) 的 `DoAL1FullLoadTiling`，能看到完整门槛：

| 条件 | 源码约束 |
| --- | --- |
| 平台能力 | `supportL0c2out` |
| dtype | A 为 FP32 |
| 转置 | A 不转置，B 转置 |
| M | M ≤ 16 |
| N | 16 < N ≤ 16 × AIC 数 |
| K | K ≥ 4096，按 512 字节对应的元素数对齐 |
| 算法取舍 | 某些小 M/N 场景若更适合确定性多核 Split-K，优先让给 Split-K |
| 容量 | 对齐后的整份 A 加 B 双缓冲和 bias，必须放得下 L1 |

容量检查并非只看 `M*K`。它按 M 对齐到 16、K 对齐到一个数据块，计算 `al1Size`，为 B 的 `16×256` 双缓冲和可选 bias 留出空间。若 A 放得下但挤掉 B 搬运空间，仍不会使用此模板。

例如对齐M=16、K=4096的FP32 A占 **256KiB**，B双缓冲占 **32KiB**，合计至少288KiB再加bias。即使真实M只有1，A的片上容量估计仍按16行对齐，并不会缩小到16KiB。K继续增大时，容量约束很快拒绝这个分支；这解释了为什么“大K”既是全载想解决的瓶颈，也会成为全载不成立的原因。这里的数字是由源码公式算出的容量，不是测得的流量或耗时。

通过后设置 `baseM=16、baseN=16、baseK=256`，`stepKa=ceil(K/256)`、`depthA1=stepKa`，表示 A 覆盖完整 K；B 的 `stepKb=1、depthB1=2`，表示 B 继续以小块双缓冲。不是 A/B 同时全载。

## 3. Kernel 怎样真正复用 A

[mat_mul_l1_full_load.h](../code/upstream/matmul/mat_mul_v3/op_kernel/mat_mul_l1_full_load.h) 的 `MatmulBaseKernelAL1FullLoad::Process` 在 AIV 上直接返回，只执行 Cube 工作。流程是：

1. 用对齐的 M/K 申请 L1 输入队列。
2. `CopyInA1` 使用 `Nd2NzParams` 将 ND A 搬到 L1 的 Cube 布局，EnQue/DeQue 保护搬运完成。
3. 在各输出块循环中，`SetTensorA(al1Local)` 一直指向同一份 L1 A；B 根据当前输出列更新 GM 地址。
4. 设置尾块形状，`Iterate/GetTensorC` 完成矩阵乘与输出；最后释放 A 队列并结束流水。

因此节省的是对同一份 A 的重复 GM→L1 搬运，代价是 A 在整个核内循环期间占据 L1。若 N 太小而没有可复用的块、K 太大导致 A 不能驻留，收益就不会成立。

本版本 [README](../code/upstream/matmul/mat_mul_v3/README.md) 明确写出 A2/A3 的 MatMulV3 不支持输入 FRACTAL_NZ。这里的 **ND→L1 NZ 转换发生在内核内部**，不能把它误读成支持外部 NZ 权重 API；950/其他硬件的规则也不能混用。

## 4. 为什么还要保留其他模板

入口 [mat_mul_v3.cpp](../code/upstream/matmul/mat_mul_v3/op_kernel/mat_mul_v3.cpp) 同时有 BL1 全载、单核 Split-K、多核 Split-K、不对齐数据转换和 Fixpipe 优化分支。小 M、大 K 不一定只需要 A 全载：如果输出 tile 太少，N 分核仍会闲置大量核，多核 Split-K 可以增加并行度，但需要部分和与归约。Host 在两者之间做选择，而不是给所有小 M shape 强行开启 full-load。

`matmul_v3_l2_cache.cpp` 还按 M/N/K 的复用与全载状态选择 L2 cache 提示。片上驻留、跨核 L2 和多核任务切分针对的是不同层次，不能把三个机制合并描述为一个“缓存优化”。

## 5. 可重复的验证设计

建议先构造 FP32、A 不转置/B 转置的 ND 输入，检查 M=8/16/17、K=4096/对齐失败的邻近值、不同 N 与真实 AIC 数。用 tiling 日志确认目标分支是否选中；只看算子名字不足以证明正在运行 AL1 全载。

候选 shape 如 M=8、N=128、K=4096；它满足可见的基本条件，但最终仍受平台能力、L1容量和 Split-K优先条件影响。它是**实验候选**，不是已验证命中模板的测试结果。相同数据和精度选项下比较普通模板与 full-load，记录 device 时间、GM→L1 读取量、Cube 利用率和数值误差。FP32/HF32精度模式要固定，避免把精度变化误算成优化收益。

[原始源码](../code/upstream/) 保存真实 Kernel、Host 和 SHA256 清单；[reproduce.md](reproduce.md) 提供固定完整工程构建路线。当前结果属于源码实战拆解，没有公开或本机实测加速比。
