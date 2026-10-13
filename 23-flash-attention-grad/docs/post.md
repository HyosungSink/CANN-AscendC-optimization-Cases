# A2/A3 FlashAttention反向：SAMEAB/BN2的基础Cube API、L1复用与CV流水

本篇研究CANN **9.0.0正式版** `FlashAttentionScoreGrad` 的A2/A3 `arch32` 实现，重点是SAMEAB与BN2两种模板的实际数据搬运、L1/L0复用和梯度流水。源码固定ops-transformer `v9.0.0`，提交`afe72144f9f2ac8441929035795db88a111b30c5`。

官方[9.0.0发布说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)报告：SAMEAB与BN2基础API优化重写，相对CANN **8.5.0**，泛化测试平均优化幅度分别为 **1%+ / 10%+**。这是模板级官方汇总，没有给出逐shape绝对耗时、完整测试集或该条对应的设备清单，不能改写为任意A2/A3工作负载的保证，也不是单个L1开关的消融结果。

[产品表](../code/upstream/attention/flash_attention_score_grad/README.md)明确支持 **A2/A3训练系列**，并把这两类推理系列标为不支持。Host注册`ascend910b`与`ascend910_93`证明实际A2/A3源码路径；本文排除950的arch35，未在本机运行NPU。

## 1. 为什么反向需要重新组织五类矩阵乘

以无Dropout、无PSE的标准Attention为例，反向需要重算或恢复概率P，再计算：

\[
dP=dYV^T,\quad dV=P^TdY,
\]

\[
dS=P\odot(dP-\delta),\quad
\delta_i=\sum_jdY_{i,j}Y_{i,j},\quad
dQ=dSK/\sqrt D,\quad dK=dS^TQ/\sqrt D.
\]

加上重算`QK^T`，包含五类GEMM。Vector负责由前向Softmax max/sum恢复P、计算Softmax梯度、处理mask/dropout及dtype转换。前向统计必须来自与当前输入相同的执行；把示例中的常量统计当成真实训练保存值，只能检查API而不能验证梯度。

## 2. SAMEAB中的三份任务记录不等于三份所有张量

[SAMEAB Kernel](../code/upstream/attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h)用`DBParams dbParams[3]`保存当前与前两轮任务信息，中间矩阵workspace则按`taskId%2`使用ping-pong槽。它们保护的是不同对象。

`Process`先让Cube计算当前任务的QK和dYV；Vector等待上一任务的`SYNC_C1_V1_FLAG`，执行概率恢复与dS；Vector的MTE3完成后通过`SYNC_V1_C2_FLAG`通知Cube继续dQ/dK/dV。流尾保留额外迭代排空仍未完成的消费者。

确定性模式另用`DTMComputeMMDqkv`写部分结果，由前两轮任务的`ComputeVecAdd`按受控顺序归约，并有相应全核同步。它不是“相同流水加一个布尔标志”，workspace、归约次序与排空长度都不同。

## 3. 基础Cube API怎样复用L1输入

`Cube1Compute`把S1/S2片段进一步切成基础M/N块。每个M子块把左矩阵载到L1；右矩阵只有在`m==0`时从GM载入，后续M子块直接复用L1中的右矩阵。

随后用`LoadDataAToL0/LoadDataBToL0 → Mmad → Fixpipe`明确控制搬运、计算、输出，而不是每个子块都重新发一个通用Matmul API。L0A/L0B采用32KiB ping-pong区，L0C采用64KiB ping-pong区；`M_MTE1`保护旧数据已经消费，`FIX_M`保护累加输出和L0C复用。

这种复用降低同一右矩阵沿M方向的GM读取，但并不意味着整个Q/K/V常驻L1。当前计算段用完后缓冲会被后续阶段复用；中间FP32结果仍经workspace交给Vector。

## 4. 长序列的L1定制模板有很窄的条件

[SAMEAB Host的ChooseL1Custom](../code/upstream/attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp)要求BF16、非TND、S1/S2都至少512、CV片段恰好512×512、无PSE/Dropout且非确定性，再按D和mask选择：

| D与额外条件 | 选择 |
| --- | --- |
| D=128 | L1定制；前两类/后三类结果使用对应NZ选择 |
| 128 < D ≤ 256 | 还要求BNSD布局与存在mask，调整后段输出布局 |
| D=64 | 还要求无mask |

例如D=128的片上分配中，Q/K各`512×128×2=128KiB`，dS相关区`512×128×2×2=256KiB`，合计512KiB。这里的缓存分配与数据生命周期紧密相连，不能增加一份“方便的临时张量”而不重算L1预算。

入口再按D模板参数选择实际实现。SAMEAB是模板名称，不能据此假定Q和K数据相同，也不能把这个定制子分支的条件当成整个算子的输入限制。

## 5. BN2模板按Batch与KV-head派发，短序列更适合

[BN2 Host](../code/upstream/attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2.cpp)的实际`IsCapable`会拒绝S1或S2 **≥1024**，还检查B×N2能否提供足够并行任务、GQA分组、RoPE、TND和确定性等条件。源码中一条错误日志文字与比较条件方向不一致，本文以实际条件为准。

在`Process`中，每核轮转处理`B×N2`任务，内部再遍历G与S1/S2块。若长序列而B×N2太小，需要借序列轴开核，Host会交给其他模板；这解释了为什么BN2不能覆盖所有Attention训练shape。

[cube_op_bn2.h](../code/upstream/attention/flash_attention_score_grad/op_kernel/arch32/basic_modules/cube_op_bn2.h)的`ComputeMM12`直接组织128×128基础子块。左矩阵只在首个N子块加载，右矩阵可在L1/L0中驻留供内层M循环使用，并通过`reuse_left/reuse_right`避免已缓存的数据再读GM。后续梯度计算沿相应布局和转置规则加载。

BN2的L1定制路径另外要求BF16、无PSE/mask/dropout、D在72–128、S1/S2大于256、非稀疏等；Host从512KiB中扣除dS与K区，按对齐后的D推算可用S1块大小。不能把SAMEAB的512×512模板参数直接套在BN2上。

## 6. 性能结论怎样复现才有意义

官方1%+/10%+是相对8.5.0的泛化平均报告，本文保存这个原始基线和口径，不反算为一个未经定义的加速倍数。要复测版本提升，应分别安装配套的8.5.0与9.0.0并固定相同数据、前向统计、dtype/layout、mask/dropout和确定性；不能只改新版本一个模板标志就称为复现官方结果。

建议分别记录B×N2、G、S1/S2、D与选中的模板，观察基础Cube API的MTE2读取、L0/Fixpipe等待和Vector重叠。S1/S2在1024门槛两侧、D=64/128/192以及B×N2不足的情况特别能说明Host选择的取舍。数值用正确前向输出与梯度参考验证dQ/dK/dV，Dropout必须复用同一mask。

[原始源码](../code/upstream/)保留实际arch32 Kernel、基础Cube模块、Host模板和许可证；[reproduce.md](reproduce.md)明确完整工程与未运行状态。本文具备官方版本级性能方向与真实内核证据，没有本机单shape实测成绩。
