# A2/A3 DynamicQuant：DeepSeek尺寸专用批量行处理与超长行两遍扫描

动态量化需要先得到一行的abs-max，再使用相同scale量化整行。这个依赖看似简单，却决定了行切分、UB容量、核数分配和是否需要重复读取。本文拆解ops-nn `quant/dynamic_quant`在A2/A3上的真实Host与Vector Kernel。

版本为正式CANN **9.0.0**，ops-nn `v9.0.0`提交`0413e303e6cd1986b592c3e039f2579ac0f10b62`。[官方配套说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)明确映射；Host def直接注册`ascend910b`和`ascend910_93`。本文不使用950 VF，也不使用310P实现。未在macOS本机运行NPU，没有公开固定版本AB耗时。

## 1. 从“每行独立”到“一次UB处理多行”

常规对称量化按token计算：

\[
z_{t,j}=x_{t,j}\,smooth_j,\quad
s_t=\max_j|z_{t,j}|/127,\quad
y_{t,j}=\operatorname{round}(z_{t,j}/s_t).
\]

smooth是可选项。每行只有一份scale；如果把一行分给多个核，必须再合并最大值并同步，可能花费比量化算术更多的时间。对能装入UB的行，本实现优先以完整行分配，再根据宽度一批处理多个token。

## 2. DeepSeek尺寸专用模板是明确的shape集合

[dynamic_quant_tiling.cpp](../code/upstream/quant/dynamic_quant/op_host/dynamic_quant_tiling.cpp) 的 `SetSpecialTilingForDs` 先于通用模板执行，要求：

- 无`group_index`，输出为INT8，不是额外输出offset的DynamicQuantV2模式。
- 行宽只取 **4096 / 5120 / 10240**。
- 行数在 **4–384**。

通过时选`TilingKey=100`，`maxHandleRowsPerUb=16384/rowLen`。因此4096宽每批最多4行、5120宽最多3行、10240宽最多1行；随后按这些UB批次数分核，而不是无论多少行都启动所有Vector核。

名称`DynamicQuantSingleRow`容易误导。它使用Host传入的`multiRowNumHeadCore`组织UB，所以这个特殊模板并不固定“一次仅处理一个token”。例如4行×4096正好是一批；这样的安排减少循环与调度开销，但也可能减少并行核数，必须结合shape测量。

该函数名直接提供了DeepSeek相关shape优化的源码证据；本文没有把它扩大为整个DeepSeek模型的性能承诺。

## 3. BF16/smooth与纯FP16选择不同计算路径

[dynamic_quant_single_row.h](../code/upstream/quant/dynamic_quant/op_kernel/dynamic_quant_single_row.h) 的`Process`对BF16或有smooth的场景使用`ProcessBf16`：转FP32、应用smooth、abs/归约，再计算逐行scale。纯FP16且无smooth则使用`ProcessFp16`，在可用的half临时区域中组织部分归约。

两条路径都在一份UB输入/临时空间内复用缓冲区域，用`ComputeReduceMax/WholeReduceMax`得到各行最大值。计算出`127/max`后广播到整行，后续执行**乘法**而非逐元素除法；输出scale则保存`max/127`。量化数据的乘因子和输出反量化scale互为倒数，不能把二者混写。

输出转换沿`CAST_RINT`等步骤处理，整体仍受具体dtype与舍入契约约束。复制这套思路时要保持全零行、非有限值、INT4打包等边界由正确的受支持路径处理，不能从常规INT8示例推断所有模式。

## 4. 通用多行模板的UB预算不是固定行数

特殊shape不命中时，`CheckMultiRowPreconditions`尝试通用`TilingKey=10`：无MoE group_index、对称量化、行宽至少128且是8倍数，并受UB预算上限限制。

它从UB大小中扣除1024字节保留空间及归约/常量缓冲；有smooth按每元素15字节预算，无smooth按11字节，再对64元素粒度对齐，且限制到`255×64`元素以满足向量repeat上限。

`DynamicQuantMultiRow`只加载一次smooth，循环中广播到多行、abs与逐行归约，再广播scale完成量化。宽度扩大或增加smooth会减少每批行数；只看输入2字节和输出1字节不足以估计它的UB容量，因为FP32 cast/abs/scale临时张量也在其中。

## 5. 超长行为何要读两遍

一整行放不下UB时，[dynamic_quant_large_shape_opt.h](../code/upstream/quant/dynamic_quant/op_kernel/dynamic_quant_large_shape_opt.h) 分K片段执行两阶段：第一遍逐段求局部最大值并更新整行max；scale确定后，第二遍重新读片段、量化并输出。

这增加HBM读取，但避免存储整行FP32中间数据，也保持一行统一scale。`CopyInByEle`和队列使下一片段读取可以与当前片段计算交错；它不能消除“先知道整行最大值”这一数学依赖。

若试图用每片段自己的最大值完成一遍量化，那会变成分块量化，改变输出scale契约，不是等价优化。对超长行，正确的问题是两遍扫描如何重叠和减少临时空间。

## 6. 应怎样形成自己的实战数据

围绕行数3/4/384/385、宽度4096/5120/10240及邻近值测专用模板边界；比较有无smooth、BF16/FP16与宽到放不下UB的shape。用tiling日志确认100/10/2或3/6等模板路径，记录实际核数和每批行数。

数值比对至少包括输出scale、随机正负数、全零/很小值、量化舍入边界。性能只统计device计算，排除输入搬运和申请内存；观察批量行模板是否降低scalar/调度时间、两遍扫描是否受HBM限制。[原始源码](../code/upstream/)、[复现](reproduce.md)和[证据记录](sources.md)提供固定版本；本文没有用模型tokens/s代替单算子验证。
