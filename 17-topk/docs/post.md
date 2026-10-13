# PTO TopK：32 元素局部排序、分级归并与 UB 双缓冲

这篇分析 **官方 cann/pto-isa 正式 v9.0.0** 的 `kernels/manual/a2a3/topk`，固定 commit `377526952e95c39f9149cb94d261eba9e394f0d4`。`version.cmake` 标注 PTO 9.0.0；CANN 正式发布配套表将这个 tag 对应到 **CANN 9.0.0**。案例目录明确支持 A2/A3，公开性能表在 A3 48 个 Vector 核测量。它不是 950 的 radix-sort TopK。[固定原始案例](https://gitcode.com/cann/pto-isa/blob/377526952e95c39f9149cb94d261eba9e394f0d4/kernels/manual/a2a3/topk/README_zh.md)。

## 1. 原始规格比“4800×1024”多一层

原始默认输入有效部分为 4800 行、1024 列，但每行 GM 的物理跨度是 **1280 列**；输出 k=1000 的 value 和 uint32 index。`gShape4` 表示有效列，`gWholeShape4` 表示行跨度。`GlobalTensor` 使用动态 shape/stride 把这两个概念分开，读取每行前 1024 个有效元素。

Kernel 固定启动 48 个 block，每块处理 100 行。`SINGLE_LOOP_ROW=2`，双缓冲的一次外循环处理两组、每组两行。源码检查总行数能被 blockDim 整除、每块行数能被四整除，且本地 buffer 总量不超过预算。这些约束决定了适用范围：不能只把 rows 换成 4801 或让 k 超过有效列而期待自动处理尾部。

## 2. 为什么先排 32 个，再归并

`SortEachGroup` 使用 `TSORT32`，把每行分成 32 元素组做局部排序，并保留值与索引的打包形式。随后 `MrgsortSingleRow` 以四路归并增加已排序段长度：`TMRGSORT` 合并到临时 Tile，`TMOV` 更新供下一层消费的输入。

这不是在 UB 中调用一个黑盒完整排序函数。每一级的 tile 长度、临时地址和已排序块的粒度都显式可见。对于无法恰好覆盖的后段，`FillMrgArray` 组织段长度，`SortTailBlock` 做收尾归并；它还将用于合并的段长度截到需要的 TopK 范围。

需要保留两个尺度：有效列是 1024，但中间 value/index 打包格式的列宽受 `TYPE_COEF=sizeof(float)/sizeof(T)` 影响。FP16 和 FP32 的中间 tile 占用不能按同一个“元素个数”计算。`ExtractDataOrIndex` 最后用不同的 `TGATHER` mask 抽出 value 和 index，写成各自的连续输出。

## 3. UB 双缓冲的关键不是复制两个数组

`InitBuffers` 显式分配两份 source、Sort32、merge 和输出 tile，以及两份 merge 临时空间。`ProcessSingleRow` 首先等待 `PIPE_V→PIPE_MTE2` 的反向事件，确认对应 source 槽位不再被上次排序使用，再发起 TLOAD。输入加载完成后通知 Vector；局部排序消费 source 之后便释放 source 槽位，下一次 MTE2 可复用，而归并和输出继续进行。

这一点比“整轮计算完再加载下一轮”更细：**source 输入可复用的时机是它最后一个消费者结束，而不是整个 TopK 输出完成**。Value 和 index 的抽取分别发出 Vector→MTE3 事件，TSTORE 各自等待对应结果。

`runTOPK` 在首轮预置反向 flag，循环结束再 drain 剩余事件。调整双缓冲时若遗漏这些初始化/收尾，可能死锁或在循环边界覆盖未消费的数据；只看稳态循环的图是不够的。

## 4. 公开性能表能说明什么

固定官方 tag 的 README 给出 A3 48 Vector 核参考数据：

| dtype | 有效行 | 有效列 / 物理跨度 | k | task duration |
|---|---:|---:|---:|---:|
| FP32 | 4800 | 1024 / 1280 | 1000 | 324.106 μs |
| FP32 | 3456 | 1024 / 1280 | 1000 | 238.819 μs |
| FP32 | 2304 | 1024 / 1280 | 1000 | 161.375 μs |
| FP16 | 4800 | 1024 / 1280 | 1008 | 326.886 μs |

同表 Vector 占比约 89%–94%，搬入/搬出占比约 10%–12%。这些引擎可以重叠，占比相加不是 100%。它们提示排序归并的 Vector 工作是值得检查的对象，但不能仅凭占比就宣称已经达到硬件带宽或理论最优。

表中没有同规格 naive / 单缓冲基线，也没有测量所用 CANN、驱动与当时 commit。因此这是绝对性能参考，不是优化加速倍数；正式 v9.0.0 的版本适用性来自 release 配套和源码，而非倒推历史测量环境。A2 支持也不等于 A2 具有同一时延。

## 5. 正确性和计时边界

原始 `gen_data.py` 生成带物理行跨度的输入，以及降序 value/升序 index 的 NumPy `lexsort` golden；Host 分别校验 values 和 indices。这比只比较输出最大值更完整。重复值的 tie 行为尤其需要额外验证，不能因为随机 FP32 输入通过就假设所有 FP16 ties 已覆盖。

默认 Host 只 launch 一次，再同步、拷回和比对，不包含可靠的重复平均计时。运行脚本的 shell 总时间包含数据生成、构建、ACL 初始化和文件 IO，不能填进 task duration 表。要对照双缓冲，先固定输入，再用 msprof 查看 TopK 设备事件；如改 Host 为重复测量，应把 warmup、计时样本、缓存状态、输入 dtype、k、stride 和 blockDim 一起保存。

这个实现适合研究“几乎保留整行”的 k≈N 场景，不保证 small-k 是最合适算法。修改 k 需要连同打包长度、临时 UB 预算、提取 mask 和 Host/golden 同步检查。本合集保留所有原始代码与构建文件，提供固定源码 wrapper；本机未做 NPU 编译或性能复测。[复现说明](reproduce.md)。
