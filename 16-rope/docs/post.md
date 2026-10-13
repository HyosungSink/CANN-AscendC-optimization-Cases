# RoPE 的算子内优化：复用广播参数，融合旋转与乘加

RoPE 的数学公式很短，但优化实现不能只把 `x*cos + rotate(x)*sin` 逐个转换成框架算子。这篇研究 **CANN ops-transformer 正式 v9.0.0** 的 `posembedding/rotary_position_embedding`，固定 commit `afe72144f9f2ac8441929035795db88a111b30c5`。算子 README 和 `ascend910b / ascend910_93` 配置明确支持 A2/A3；正式配套表将 ops-transformer v9.0.0 对应到 CANN 9.0.0。源码同时有其他硬件实现，本篇只分析 A2/A3 的 Ascend C 路径，不使用 arch35/950 内核。[固定算子文档](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/posembedding/rotary_position_embedding/README.md)。

## 1. 先确定旋转模式和广播关系

`mode=0` 为 rotate-half：把 D 维切成前后两半，再组合 `[-x2,x1]`。`mode=1` 为 interleaved：相邻偶数、奇数元素交换位置，其中一个取负。两种模式都进行乘 cos、乘 sin、再相加，但重排方式不同。

A2/A3 文档对布局和广播有明确约束：D 是偶数且小于 896；half 模式支持 BNSD/BSND/SBND 的若干广播组合；interleave 的 cos/sin 广播形式更窄，且 B×N 有限制。不能把 950 的任意广播、quarter 模式等能力移植为本例的条件。一个高质量 benchmark 必须记录 mode、布局、广播形状、dtype，不能只记录总元素数。

## 2. Half 路径的优化重点：按广播范围复用 cos/sin

阅读 `rotate_half_bf16.h` 的 `Process`、`SingleStepProcess`、`RB1sdSingleStepProcessBf16` 与 `BndProcessBf16`。`Process` 根据 tiling 的布局类别选择普通、B1SD 广播或 BND 等处理路径，而不是为所有输入强行展开 cos/sin 到完整 X 大小。

`SingleStepProcess` 先搬入当前 sin/cos 块并转为 FP32，准备正负号，然后遍历对应的广播维度，对多块 X 重复使用这些参数。`BndProcessBf16` 更直接：在循环外加载一份 cos/sin，再用于后续 X 块。这样复用发生在算子内部，不产生一份扩展到 `[B,N,S,D]` 的全局广播张量。收益来源是减少重复参数搬运和中间存储；不是简单把某个循环删掉。

## 3. 融合不是只减少启动，也减少了中间 GM 往返

`RotateHalfBase::XNewCopy` 负责半维旋转的数据组织，`SinCompute` 为相应半维设置符号，`ComputeInner` 执行两次 Mul 和一次 Add。所有这些在已搬入的本地张量上完成；最终只写输出 Y。相比框架组合的 slice/concat/negative/multiply/add，它将重排与算术放在同一 Kernel 中。

FP16/BF16 路径采用 `RotateHalfBf16<OriT,float>`：X、sin、cos 转 FP32 参与计算，输出再 Cast 回原 dtype。精度和 UB 预算因此耦合：本地临时张量比输入更大，不能把 FP32 计算缓冲忽略后再盲目增加每块行数。Host `rope_rotate_half_tiling.cpp` 按 dtype、双缓冲和对齐长度计算可处理量，这就是为什么“相同元素数量”不保证相同 tiling。

## 4. 队列与尾块要一起看

Half 实现为 X/cos/sin 输入和 Y 输出建立双缓冲队列。数据通过 AllocTensor、EnQue、DeQue、FreeTensor 的依赖进入计算；`ubLoop / ubLast` 分别处理完整工作块和尾部。布局不同还会改变 `DataCopy` 的 stride：BSND/SBND 不能按 BNSD 连续块的地址直接搬运。

非对齐 D 的路径使用 `DataCopyPad`，为两个半维分别处理 padding；写回只覆盖有效数据。用 D=128 的默认小样例只能验证一个对齐分支，无法覆盖 D=126 的 padding、跨 UB 尾块或广播复用。调优时若更改 tile、队列深度或跨步地址，需要验证这些分支。

## 5. Interleaved 路径：用 Gather 完成相邻配对重排

`rotate_interleaved_split_bs.h` 的 `Process` 预先生成 gather offsets；`Compute` 搬入 X 和广播 cos，计算 X*cos，再在本地用 Gather 交换配对元素，乘 sin 并合并结果。`CopyInCos / CopyInSin` 先取当前位置的一份 D 参数，再通过 `BroadCastTriToB1nd` 扩展到当前本地处理块。这样避免先在 GM 做一个完整广播。

BF16 的 `ComputeCastFp32` 明确增加了转换和 FP32 缓冲。S、BS、BSN 的多个 kernel，以及相应 pad 版本，说明实现根据工作量与布局选择分核/搬运方式。不能把某个 SplitBS 的 tile 常量解释为所有 RoPE 输入的最优选择。

## 6. 性能证据与上板验证

这是一个 **正式版本的具体优化源码案例**，目前没有核实到同时标注 CANN 9.0.0、A2/A3、shape 与基线的公开 RoPE 加速表。本篇不提供虚构耗时，也不把默认 `{1,1,1,128}` 的接口样例称为性能 benchmark。

本地保留完整 A2/A3 核心实现、Host Tiling、ACLLN 调用样例与测试。建议先运行固定版本样例，再修改样例规格，对 `(B,N,S,D)=(2,8,257,128)`、D=126、mode 0/1 和合法广播分别测试。黄金值应按旋转模式用 FP32 公式计算，并检查 BF16/FP16 输出误差。再用 msprof 比较设备 Kernel duration，而不是把 ACL 初始化、Host→Device、打印和释放资源都算成算子时间。

真正的对照应固定同一形状、布局、dtype 与广播，比较融合 Kernel 和显式拆解公式。只有实际运行后才能判断广播复用、FP32 转换成本以及双缓冲是否得到收益。本机没有 NPU/CANN，已完成源文件一致性和 wrapper 静态检查；设备编译、正确性和性能仍待上板。[复现步骤](reproduce.md)。
