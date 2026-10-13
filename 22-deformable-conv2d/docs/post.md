# A2/A3 可变形卷积：复用采样坐标、合并相邻访存，再接 Cube 矩阵乘

本篇研究 CANN **9.0.0 正式版** ops-nn `conv/deformable_conv2d` 的实际 A2/A3 内核：offset/mask → 四邻点双线性插值 → 卷积输入重排 → Cube GEMM。它是可变形卷积实战，**不冒充普通 Conv2DV2**。同一正式标签中的 Conv2DV2 / ExtendConv2D 明确不支持 A2/A3，因此没有拿它们的 950 内核来充数。

源码固定 `v9.0.0` 提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`，[正式配套说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)明确配套 CANN 9.0.0。算子 [README](../code/upstream/conv/deformable_conv2d/README.md) 支持 A2/A3，排除950；Host实际注册`ascend910b`、`ascend910_93`。没有找到该版本的公开A/B单算子耗时，本机也没有NPU。

## 1. 算子内部真正耗时间的是什么

普通卷积访问规则的滑动窗口；可变形卷积先为每个窗口点添加二维offset，再按坐标找到四个邻点，用双线性权重插值并乘调制mask。固定版本的aclnn检查要求`modulated=true`。这会产生离散HBM读取和大量坐标计算，单纯提高GEMM效率无法解决整条Kernel的瓶颈。

[主循环](../code/upstream/conv/deformable_conv2d/op_kernel/deformable_conv2d_base.h)按`batch×输出高度`分给Vector任务，每个任务逐输出宽度和deformable group处理采样。输出一整行所需的插值数据准备完成后，`Conv2d`按卷积分组执行矩阵乘。它不是取消所有中间数据：插值结果同时写到可选`deform_out`布局与用于GEMM的GM workspace。

必须区分两个group：`deformableGroups`决定offset和插值参数按哪些输入通道共享，`groups`决定权重与卷积输入的分组GEMM。二者不能互换。

## 2. 固定UB预算如何限制滑动宽度

Host的 [SplitCore](../code/upstream/conv/deformable_conv2d/op_host/deformable_conv2d_tiling.cpp)以`MAX_KERNEL_SIZE=2048`控制一批采样坐标数。若`deformableGroups×Kh×Kw > 2048`，宽度步长设为1，并切deformable group；否则把这个乘积按数据块对齐，再计算能同时处理多少输出宽度。

Kernel定义`DATA_COUNT=2048`、`MAX_INPUT_SIZE=1024`。FP32时，坐标、offset、floor/ceil、四份权重、四份索引和mask约占 **138KiB UB**；输入邻点、插值结果、gather索引等再占 **44KiB**，合计约 **182KiB**。这些是由`InitBuffer`公式计算的缓冲预算，不是测得的内存带宽。

这解释了为什么不能随意扩大输出宽度tile：采样位置数随`宽度×group×Kh×Kw`增长，同时四邻点数据与FP32临时空间都要驻留。Host还把`batch×outH`按商与余数分配，余数核多处理一行，避免整行任务数量不能整除核数时产生明显不均衡。

## 3. 标准坐标不用每个tile重算

`CalculateStandard`用scalar循环生成卷积基础坐标；首个tile或尾部tile形状改变时执行它。后续同形tile用`AdjustStandard`，以Vector `Adds`加上宽度/高度位移。

这样省掉了重复的`输出宽度×group×Kh×Kw`嵌套标量索引生成。offset本身仍然从GM读取并叠加；复用的是规则基础坐标，不是可变形offset。

`CopyInOffset`分三路读x/y偏移及mask。数据块不对齐时，DMA产生的填充间隙用`GatherMask`压紧，避免把padding当成真实采样点。`CalculateWeight/ProcessZero`同时准备四邻点索引和双线性权重，对越界位置设索引-1、权重0，减少后续无效输入读取。

## 4. 小通道路径为什么能减少DMA次数

Kernel在`deformableGroups=1`、输入C按32字节块对齐、`Kh×Kw×C ≤ 1024`等条件下选`BilinearInterpSmallC`，并进一步限制当前宽度批次能装入临时区域。

NHWC内部布局使同一行相邻两个像素的C通道连续。四邻点都在边界内时，`CopyInSmallC`用一次二维`DataCopyPad`读两行，每行包含左右两个邻点；常规路径则分别读四份邻点。边界不完整时按有效的上/下两行拆分，不对越界位置发读取。

采样批次读完后，`GatherMask`把各邻点的通道数据拆出，广播每个采样点的双线性权重，再用Vector乘加完成多个点的插值。通道大时则按最多1024个元素分块，走`ComputeEntireC → CopyInAndCalculate → CopyOut`，每块用四次有效邻点读取与`Axpy`累加。这是明确的小C专用优化，不代表所有shape都能合并成一次DMA。

## 5. 插值布局与GEMM布局不同

可选deform输出按空间展开排列；GEMM workspace则按`[batch,outH,groups,outW,Kh×Kw,C/group]`组织。小C路径中，多个卷积分组通过`Gather`重排到对应workspace片段；常规路径在`CopyOut`中计算每组偏移。

每个输出行、每组的矩阵乘形状为：

```text
weight: [Cout/group, Kh×Kw×Cin/group]
sampled input: [outW, Kh×Kw×Cin/group]，作为转置B
output: [Cout/group, outW]
```

`Conv2d`调用`matmulObj.SetTensorA/SetTensorB/IterateAll`完成Cube工作。本文保存了实际采样、插值和重排Kernel，不只保存这个Matmul调用。

workspace预算为`batch×outH×outW×Kh×Kw×Cin×元素字节 + 16MiB`。融合采样与GEMM减少独立采样算子提交，但完整插值输入仍会落GM；它既有读取成本，也为跨Vector/Cube的布局与数据依赖提供存储。

## 6. aclnn接口路径的成本不能藏起来

固定版本的`aclnn_deformable_conv2d.cpp`在A2/A3上先把输入/offset/weight转换到内部布局；若外部dtype为FP16/BF16，还会转到FP32后执行Kernel。bias通过API后处理的`Add`完成，输出再Cast/Transpose/ViewCopy。

因此不能宣称“FP16 aclnn接口全程低精度”或“bias也完全融合进本Kernel”。计时应把核心DeformableConv2d时间与整条aclnn执行链分列；否则布局转换、精度转换和bias操作会混进优化结果。

API即使不要求返回`deformOutOptional`，L0启动层仍分配内部deformOut，当前Kernel仍写出它。关闭外部可选输出不等于省掉这一份内部HBM流量；若要消除，必须修改内部数据路径并重新验证，而不能只改变API参数。

## 7. 可以验证的优化实验

先以FP32、NHWC内部输入、`deformableGroups=1`比较C按8对齐的小C路径与不对齐/大C常规路径；再改变卷积group、offset造成的越界比例、Kh/Kw与输出宽度。确认tiling和`calW`分支，并观察DMA次数、scalar基础坐标生成和Vector插值时间。

数值参考应对照相同offset/mask和padding的双线性插值，再进行分组卷积；零offset、mask=1应退化到规则采样的结果。公开AB数字缺失，本篇不填加速倍数。[原始源码](../code/upstream/)、[复现流程](reproduce.md)和[证据](sources.md)提供固定版本与未运行边界。
