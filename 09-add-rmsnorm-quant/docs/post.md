# AddRmsNormQuant 的 UB 预算与量化融合

AddRmsNormQuant 将残差相加、RMS 归一化和 INT8 量化放在一个算子中。真正值得研究的地方，是融合后不同形状如何选择 Normal、SingleN 或 SplitD，以及 scale、zero point 和额外输出如何反过来改变 UB 工作集。仅说“融合减少访存”无法解释这些分支。

本文分析 `cann/ops-nn` 正式 `v9.0.0` 的提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`，保留 A2/A3 的配置与非 RegBase Kernel。该版本产品表明确支持 A2/A3；不使用 `arch35` 的 950 专属性能路径。[正式配套](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)、[算子目录](https://gitcode.com/cann/ops-nn/tree/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/add_rms_norm_quant)

## 先保留精度路径再讨论融合

基本计算是 `x=x1+x2`，然后 `rstd=rsqrt(mean(x²)+eps)`，归一化结果乘 gamma，按量化 scale/zero point 转成 INT8。scale 的使用受 divMode 控制，不能把所有场景都改写成乘 scale。算子还可输出残差和 x；需要该输出时，融合不会消除这一份 GM 写出。

`CopyInAndAdd` 对 FP16 与 BF16 有不同的精度路径。FP16 分支先在输入精度相加，再 Cast 为 FP32；BF16 分支先将两个输入转换到 FP32 再相加，并按路径将残差输出转换回 BF16。带可选项的分支还有针对舍入行为的转换。因此 CPU 参考若从头到尾使用 FP32，再只对最后结果 Round，可能与真实融合算子的中间舍入不一致。量化阈值附近更容易体现这一差别。[残差计算](../code/upstream/norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant_split_d.h)

## 每核只有一行时优先 SingleN

`Tiling4AddRmsNormQuantNotRegbase` 先按 N 行平均分核，`blockFactor=ceil(N/coreNum)`。`CalcModeAndUbFactor` 在每核一行、不是 310P 且 D 能放入单行预算时选择 SingleN；普通行宽不超过 Normal 预算则选择 Normal，否则选择 SplitD。

默认常量分别是 `singleUbFactor=12224`、`normalUbFactor=8192`，低精度 SplitD 的初始分片参考为 7680，FP32 为 4096。这些数字没有独立于输入条件的普适意义：Host 会根据可选缓冲进一步重算。

SingleN 的 `ProcessFp16`/`ProcessBf16` 针对一行组织 x1、x2、FP32 归约、gamma 和量化临时量。代码直接将已消费的 x2 区域复用为 gamma，后续 scale 与 INT8 输出也利用临时 tensor 的 ReinterpretCast。重点是同时存活的存储更少，允许一行保留在 UB 内完成；它没有把 D 拆到多个 AIV 共同求和。[SingleN Kernel](../code/upstream/norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant_single_n.h)

## 可选项改变容量而不只是多做几条算术

`optionalUbNum_0` 与 `optionalUbNum_1` 是 Host 对附加缓冲的预算。Normal 的可用元素数从 `(usableUB-256)/(20+optionalUbNum_0)` 得到，SplitD 使用 `(usableUB-2560)/(22+optionalUbNum_0)`，SingleN 使用 `usableUB/(16+optionalUbNum_1)`；结果再按 32 元素对齐。开启第二路 scale/output 或 zero point 会增加分母，可能使同一个 D 从 SingleN/Normal 切到 SplitD。

这是融合算子常见的隐性性能边界：API 上只是多了一个可选输出，Kernel 却需要换数据流。评测必须保存所有可选输入与输出配置；只报告 N、D、dtype 不足以解释结果。[动态预算](../code/upstream/norm/add_rms_norm_quant/op_host/add_rms_norm_quant_tiling.cpp)

## SplitD 的两遍与残差重算

`KernelAddRmsNormQuantSplitD::SubProcess` 第一遍依次调用 `ComputeFormer`，搬入 x1/x2、相加、平方并累计整行统计；随后 `ComputeRstd`。第二遍由 `ComputeLatter` 搬入该列片的 gamma、scale 等参数，再次通过 `CopyInAndAdd` 得到残差，生成归一化量化输出。

这里用重复读取与重复加法换取较小 UB，代码并没有把整行残差写到一个独立 FP32 workspace 再读回。对于启用残差 x 输出的模板，`CopyInAndAdd` 本身还包含 x 的写出，要检查两遍调用的实际行为，不能根据融合公式声称“x 完全不落 GM”。

归约先产生块内 FP32 统计，`BlockReduceSumFP32` 再合并各块；量化使用 `RoundFloat2Int8` 而不是随意 `Cast<int8>`。scale 为标量的某些 divMode 分支通过 `1/scale` 与 `Muls` 执行，向量 scale 则通过已有 `doScales` 路径处理。[SplitD 与量化工具](../code/upstream/norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant_split_d.h)、[基础函数](../code/upstream/norm/add_rms_norm_quant/op_kernel/add_rms_norm_quant_base.h)

## 可以做的两组具体对照

第一组在 N 不超过 AIV 数时，固定 FP16、D=8192，比较原始 SingleN 与独立副本里禁用该选择后得到的 Normal。参数和额外输出必须一致；这个 shape 本身能进入 Normal，避免将“强制不适合的模板”当基线。观察单行 UB 复用是否减少参数搬运和 queue 周转。

第二组固定 N 与 D，在不改变真实量化语义的前提下比较等价的标量 scale 与按 D 广播的向量 scale；先检查两种输入是否被 API 允许以及零点配置是否相同。记录 Host 预算、Tiling key 和计时，才能判断差异来自标量运算还是容量切换。第二路输出不等价于第一路输出，不能用关闭它作为相同语义的速度对照。

INT8 输出要重点覆盖舍入点、饱和边界、零输入、负值、非对齐 D 和 BF16 中间转换；残差 x 也必须独立检查。源码同时有 AddRmsNormQuantV2 的可选输入 key 编码，本文复现入口是原 `add_rms_norm_quant`，不能将 V2 的 key 与旧模板入口混用。具体编译和样例见 [复现说明](reproduce.md)。本文保存了实际实现与分支证据，没有声称本机测得加速。
