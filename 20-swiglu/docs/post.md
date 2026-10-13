# SwiGLU 在 A2 和 A3 上怎样选择缓冲与搬运块

SwiGLU 将一个输入张量沿指定轴分成 A 和 B，计算 `A / (1 + exp(-A)) × B`。本案例研究 CANN 9.0.0 正式配套 `ops-nn v9.0.0` 的 A2/A3 Vector 实现。它的优化重点是输入两半的跨行搬运、FP32 临时空间复用，以及 Host 在单缓冲和双缓冲之间的选择。

固定 commit 为 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。产品表和 `swi_glu_def.cpp` 均包含 `ascend910b` 与 `ascend910_93`，本文使用非 arch35 路径。[正式配套](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)、[固定算子目录](https://gitcode.com/cann/ops-nn/tree/0413e303e6cd1986b592c3e039f2579ac0f10b62/activation/swi_glu)

## 把分轴问题变成二维搬运

`SetTotalShape` 将 split 轴之前的维度乘积作为 rowLen，从 split 轴开始的连续区域一分为二，得到 colLen。对末轴分割的 `[M,2D]` 输入，就是 M 行、每半行 D 个元素。A 和 B 在 GM 中交织，而输出为紧凑的 `[M,D]`。

`CaclOneTileCopyParam` 给 A/B 的搬运设置 `blockCount=tileRows`、`blockLen=tileCols×inputDTypeLen/blockUnit`，跨行 stride 依据输入 `2D` 的行距计算；输出 stride 则依据 D。这样可以一次搬运多行的某个列块，省去每行单独发起 DMA。offset、blockLen 和 stride 的单位随对齐分支变化，不能把元素数直接写进按 32 字节计数的参数。[布局与搬运参数](../code/upstream/activation/swi_glu/op_kernel/glu_tiling_kernel.hpp)

普通、行尾、列尾和行列同时到尾部的 tile 分别预先生成 Copy 参数。32 字节不对齐时走 `DataCopyPad`，UB 中的计算长度向上对齐，但 GM 读写仍按实际有效列数控制。算子因此能处理尾部，而无需在热点 Vector 指令中逐元素判断全局边界。

## 同时优化 tile 形状和核间任务数

Host 默认按 512 字节对应的元素数打包列块，在 tile 容量约束内寻找 baseRowLen 与 baseColLen。`SaveOptBaseShape` 同时估算总 tile 数、使用核数和每核 tile 数。Kernel 再从 `GetBlockIdx()` 开始，以 blockNum 为步长轮转处理 tile，避免把全部行固定分给少数核。

当 DMA stride 超过接口的 16 位字段可表示范围时，Host 强制 baseRowLen=1，放弃多行一次搬运。这个限制说明，多行合并不能无限增加：索引形状、缓冲容量和指令字段必须同时满足。[Tiling 计算](../code/upstream/activation/swi_glu/op_host/swi_glu_tiling.cpp)

## FP16 和 BF16 用 FP32 计算并复用中间空间

`swi_glu.cpp` 为 FP16/BF16 选择 `SwigluVectorBF16<输入类型,float,输出类型,bufferNum>`，FP32 则走 `SwigluVector`。类名包含 BF16，但 FP16 同样使用这个计算实现。

低精度路径先将 A 转成 FP32，执行指数和除法。得到 `A/(1+exp(-A))` 后，原来保存 A 的 `inputTempBuffer` 已经可以用于 B 的 Cast；最后只将结果转回输出类型。初始化阶段把 beta 设为 -1，不能仅看父类默认值 1 就认定公式符号错误。AICore 220 的最终 Cast 使用 `CAST_RINT`。[低精度计算与缓冲复用](../code/upstream/activation/swi_glu/op_kernel/swi_glu_bf16.hpp)、[初始化和搬运](../code/upstream/activation/swi_glu/op_kernel/swi_glu_single.hpp)

这种复用避免为 A 和 B 各分配一份始终同时存活的 FP32 临时区域；它依赖 A 的最后一次消费已经结束。移动 `FreeTensor` 或提前装入 B 时，应重新核对这条生命周期。

## 双缓冲增加流水空间也压缩单 tile 容量

输入 A、输入 B、输出 C 共三个队列。低精度路径另有两个 FP32 临时 buffer。按 Host 的预算公式，每个输出元素需要的 UB 字节数为：

| 路径 | 单缓冲预算 | 双缓冲预算 |
| --- | ---: | ---: |
| FP16 或 BF16 | `3×2 + 2×4 = 14` | `2×3×2 + 2×4 = 20` |
| FP32 | `3×4 + 1×4 = 16` | `2×3×4 + 1×4 = 28` |

这些是源码的容量预算，不包括任意 SDK 额外保留空间；实际 UB 大小由平台查询得到。更多队列槽提供搬运、计算与写出的交错空间，却使每次可容纳的 tile 变小。队列深度为 2 也不等于任意 shape 都会获得两倍吞吐。

## 小工作量主动关闭 Double Buffer

`CalcTiling` 先求双缓冲方案。只有当该方案每核恰好需要两个 tile 时，才进一步计算单缓冲方案。如果单缓冲能变成每核一个 tile，而且使用的核数没有减少，就设置 `isDoubleBuffer=0`。

这是一条具体的性能选择：用更大 tile 少执行一次 CopyIn/Compute/CopyOut，而不是为仅两轮的工作维持更小的双缓冲切分。入口据此实例化 bufferNum=1 或 2 的模板。由此可以把“增大 tile”和“增加并行缓冲”作为两个相互制约的变量分析，而非默认双缓冲总是最优。

## 有意义的对照实验

先对 FP16/BF16/FP32、不同 M、D，以及 D=543 这类非 32 字节对齐形状做数值校验。参照实现用 FP32 完成指数、门控后再转输出类型；同时记录实际误差与上游精度契约，覆盖负大值、小值和尾块。

观察 tiling 日志中的 `isDoubleBuffer`、baseRowLen、baseColLen 和使用核数，寻找自然落到上述单缓冲选择条件的 shape。消融时改变 Host 的选择条件，并重新构建生成匹配的 Tiling 和 Kernel，不能只修改下发结构中的缓冲标志。再在相同核数、dtype、布局下比较 device duration、tile 数和 MTE2/Vector/MTE3 重叠。

该正式快照未提供优化前后耗时表，本文以可核对的生产源码为证据，不给定量速度承诺。代码和构建入口见 [复现说明](reproduce.md)，原始出处见 [来源记录](sources.md)。
