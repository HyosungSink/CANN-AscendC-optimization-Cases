# INT8 GEMM 的 per-token 反量化：提前算 scale 外积，再消费 Cube 输出

本篇分析 ops-nn `quant_batch_matmul_v3` 的 **A2/A3 `BmmDequantPertokenOpt`** 实现。它融合 INT8 GEMM 与反量化，尤其针对小 M 的增量 shape，把 Vector 能提前执行的 scale 组合放在 Cube 结果到达之前。

版本固定 CANN 9.0.0 正式 `v9.0.0` 标签、提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。Host注册A2的`ascend910b`与A3的`ascend910_93`，Kernel入口用A2/A3编译架构条件包含此实现。官方[配套表](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)明确标签与CANN关系。没有公开固定版本A/B耗时，本机没有NPU。

## 1. 反量化不是一个标量乘法

逐token激活scale沿M变化，逐通道权重scale沿N变化。INT32累加结果 `C_int32[M,N]` 的恢复过程是：

\[
Y_{m,n}=C^{int32}_{m,n}\,s^x_m\,s^w_n+b_{m,n}.
\]

若等Cube完整产出后，再把两种scale广播、相乘、读结果并乘一次，Vector的准备阶段也会落到关键路径。本实现提前生成块内 `s^x ⊗ s^w`，待结果到达时主要执行INT32→FP32、乘这个scale矩阵、处理bias与输出转换。

## 2. Host 为什么优先小M，又限制workspace

[quant_batch_matmul_v3_tiling.cpp](../code/upstream/matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp) 的 `ProcessMSmall` 计算每核缓冲 `baseM × align(singleCoreN,baseN)` 的INT32结果，乘AIC数得到workspace需求。

逐token优化要求对应workspace **小于50MiB**。典型入口条件包括M不超过baseM，以及平台/布局等`CheckSupportConditionQbmm`条件；它并非所有per-token输入都能无条件命中。外部MC2 tiling场景还会排除此opt模板。

选择per-token opt后，将`tbeTiling.m_al1/n_bl1`设为1，按base块推进CV消费。更大M会引入更多M循环和scale矩阵空间，继续使用小M模板不一定合理。

`CalcPertokenOptUbTiling`明确给UB预留：输入/输出双缓冲、INT32→FP32缓冲、`baseM×baseN`的FP32 scale矩阵、逐token scale、逐通道scale及非INT32 bias。剩余UB再算`ubCalcM`，并限制它不超过baseM和真实M。这里的scale预计算并非免费，它会减少可用于数据块的UB。

如果实际tiling给出baseM=16、baseN=256，仅scale外积就占16KiB；baseM变为128时这份缓冲扩大到128KiB，输入输出双缓冲还没有计入。这个由参数公式推算的例子说明，预计算可以减少等待后的算术，却可能迫使数据按更少行分批。测量时必须连同ubCalcM记录，不能只比较矩阵乘的M/N/K。

## 3. 在等待Cube前做什么

[quant_batch_matmul_v3_pertoken_opt.h](../code/upstream/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_opt.h) 中，`PertokenGm2Ub`按M块搬一次逐token scale；`PreUbProcess`调用`BroadCastMN`广播到M/N二维，再乘当前N片段的通道scale。per-tensor权重scale则直接用标量乘；BF16 scale先转FP32。

这些操作不依赖INT32 GEMM结果，因此可先进行。只有当前Vector子核`GetSubBlockIdx()==0`实际执行这份Vector后处理，另一个子核参与必要的事件同步；不能把CV 1:2编排直接解释成“两份Vector都平均处理数据”。

Cube在`MMCompute`里`Iterate/GetTensorC`写入GM INT32 workspace，并通过`NotifyEvent<PIPE_FIX>(0x4)`通知就绪。Vector完成scale准备后`WaitEvent(0x4)`，再执行`PostUbProcess`读取结果，避免提前读取未完成的Fixpipe输出。

## 4. Bias的dtype会改变放置位置

INT32 bias可以随Cube累加，在`MMCompute`的`mm.SetBias`路径处理；浮点/BF16/FP16 bias则在Vector反量化结果后相加。二者的量纲不同，不能为了“减少一次加法”把浮点bias直接塞到INT32矩阵乘前。

`PostUbProcess`按`ubCalcM`分段读GM，处理M尾块和N对齐；将INT32转换为FP32，乘准备好的scale矩阵，再转换到最终输出类型。GM INT32中间结果仍在，优化改变的是计算重叠与后处理融合，而非取消所有中间HBM访问。

## 5. 快Cube也可能让同步计数溢出

`MMDequantCompute`把N循环分成每组最多 **15** 次通知。源码说明小K时Cube可能连续产出，硬同步计数器范围为0–15，若无限通知可能溢出。组之间通过事件`0x5`等待Vector消费进度，再继续生产。

这是一项具体的流水正确性设计：优化小K计算后，生产者速度变快，反而更容易暴露同步计数风险。因此不能只增加输出列循环数量而保留无限通知，也不能把反向事件当成无用等待删除。

因此N块数从15变为16时，同步组织也会变化：前一组的消费进度需要反馈后才能推进下一组。性能曲线在这一附近出现转折并不一定是Cube算术效率变化，也可能是生产消费节奏和事件边界导致。应结合等待区间判断，而非用总时间猜测。

## 6. 可验证的实验

先固定INT8输入、FP32逐token scale、受支持的权重scale与BF16/FP16输出，核对实际tilingkey。围绕M≤baseM门槛、workspace50MiB门槛、N块数15/16、bias dtype和K长短进行实验；观察准备scale是否隐藏在Cube计算时间内。

对照普通per-token模板时必须保持量化参数与舍入一致。记录单算子device时间、Vector等待0x4/0x5时间、workspace与数值误差，区分“提交开销下降”和“CV执行重叠”。公开AB数字缺失，所以[复现文档](reproduce.md)不填虚构成绩；[原始源码](../code/upstream/)保存可逐行核对的原始实现。
