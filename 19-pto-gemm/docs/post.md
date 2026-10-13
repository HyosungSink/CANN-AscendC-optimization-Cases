# PTO 高性能 GEMM：24 核二维切分、stepK 缓存与两级 ping/pong

这篇分析 **官方 PTO v9.0.0 / CANN 9.0.0** 的 `kernels/manual/a2a3/gemm_performance`，固定 commit `377526952e95c39f9149cb94d261eba9e394f0d4`。它是纯 GEMM 内核，区别于本合集的 GEMM+AllReduce 通信融合案例。支持 A2/A3；公开参考表在 **A3 24 Cube 核**测量。[固定原始案例](https://gitcode.com/cann/pto-isa/blob/377526952e95c39f9149cb94d261eba9e394f0d4/kernels/manual/a2a3/gemm_performance/README_zh.md)。

## 1. 把工作量映射到核：不是只增加 blockDim

默认 `M=K=N=6144`，FP16 输入、FP32 输出。`InitGMOffsets` 用 `block_idx%4` 选择 M 分区，用 `block_idx/4` 选择 N 分区；形成 4×6 的网格，每核负责 `1536×1024` 的输出，K=6144 留在核内累加。

因此 24 核之间没有对同一 C 元素的归约和写冲突。A、B、C 的 GM 起点依据全局 stride 计算，B 的物理布局还需要跟随 `GlobalTensor` 的布局定义和 golden 转置设置。不能在只改 Host 的 M/N 常量后继续使用原来的 singleCoreM/N、blockDim 与输入文件。

## 2. Base block：128×256×64 的容量理由

默认 `baseM=128,baseN=256,baseK=64`。FP16 的 L0A micro tile 是 16 KiB，L0B micro tile 是 32 KiB；L0A/L0B 各自以 `0` 与 `32768` 两个槽位做 ping/pong。B 刚好达到单槽预算，扩大 N 或 K 会立即改变是否装得下，而不是一个免费的 FLOPS 增益。

这个 tile 也使 FP32 C 的每行宽度为 256×4=1024 字节，便于输出对齐。计算中 first K iteration 用 `TMATMUL` 初始化累加，其余 K iteration 用 `TMATMUL_ACC`。直到 K 完成后 `StoreResult` 才从 Acc 写回，避免每个 K 片反复往 GM 写中间结果。

## 3. stepK=4：先搬 panel，再从 L1 抽 micro tile

`ProcessKIteration` 每四个 K 迭代执行一次 TLOAD，将 A 的 `baseM×(baseK*stepKa)` 和 B 的 `(baseK*stepKb)×baseN` 搬到 L1。随后通过 TEXTRACT 取当前 64 深度的 micro tile 送入 L0A/L0B。

这样把一次搬运粒度扩大为 256 深度，减少小 DMA 的下发，并把后续抽取与计算放入流水。它不是跨所有 M/N tile 自动永久缓存 A/B；需要跟着 i/j/k 的循环看实际复用范围。固定逻辑以 stepKa 的周期同时加载 A/B，默认 stepKa=stepKb=4。把两者单独改成不同值之前，需要检查事件发布与 B 的索引是否仍正确，不能将它们当成两个完全独立的调参旋钮。

## 4. 两个 buffer index 管理两种不同生命周期

`mte2DBFlag` 控制 L1 staging 的 ping/pong；`mte1DBFlag` 控制 L0A/L0B 的 ping/pong。两者不能合并成一个“轮次奇偶”而省掉事件。

- L1 槽位要等 TEXTRACT 的最后一个消费者结束，才允许下一次 TLOAD 覆盖。
- L0A/L0B 槽位要等当前 TMATMUL 完成，才允许下一次 TEXTRACT 覆盖。
- TEXTRACT 完成后向 Cube 发布事件，矩阵乘完成再向 MTE1 返回槽位。
- 输出 TSTORE 完成后，FIX→Cube 事件保护 Acc 的再次使用。

`InitSyncFlags / WaitSyncFlags` 给出首轮启动与末轮 drain。重构软件流水时，判断每个 wait 对应哪一个 buffer 的最后使用者，比单纯减少 wait 数量更可靠。

## 5. 原始性能表与它的实际含义

正式 tag 的 README 保存 A3 24 核、FP16→FP32 的参考结果：

| M=K=N | TMATMUL 占比 | TEXTRACT 占比 | TLOAD 占比 | TSTORE 占比 | 时间 |
|---|---:|---:|---:|---:|---:|
| 1536 | 54.5% | 42.2% | 72.2% | 7.7% | 0.0388 ms |
| 3072 | 79.0% | 62.0% | 90.9% | 5.8% | 0.2067 ms |
| 6144 | 86.7% | 68.1% | 95.2% | 3.1% | 1.5060 ms |
| 7680 | 80.6% | 63.0% | 98.4% | 2.4% | 3.1680 ms |

这些执行单元可重叠，占比相加超过 100% 是正常的。最大的规格中，TLOAD 很忙且 Cube 占比下降，提示应检查供数与重叠；但利用率不是实测带宽值，单靠 TLOAD=98.4% 不能断言 HBM 已达理论峰值。

原表没有测量 CANN、驱动、当时 commit，也没有关闭双缓冲或其他 GEMM 基线。因此它是算子级绝对性能和瓶颈观察，不是当前固定源码在 CANN 9.0.0 的已复现成绩，也不是可直接引用的加速倍数。正式配套证明版本范围，实际设备复现必须单独完成。

## 6. 如何在这个实现上做有效对照

上游 Host 和生成器都固定 6144³。先生成对应 FP16 输入与 FP32 golden，再编译运行默认配置，要求 `ResultCmp` 通过。Host 只有一次 launch、同步和拷回，未预热和重复平均计时；`run.sh` 的总耗时还包含构建、IO 和 ACL 初始化，不能拿它对比表中的 ms。

使用 msprof 读取 GEMM 设备事件，并将默认 stepK=4 与合法的同形状变体比较。对 base tile 的任何改动必须重新检查 L1/L0/Acc footprint、GM stride、完整 K 的初始化与累加和写回同步。不同大小需要同步更改 Kernel 模板实例、Host、生成器和 24 核分区，不能只替换一个文件中的常量。

这个案例不是一个 16×16 的教学矩阵乘；它包含可直接阅读的完整 24 核 Kernel、两级同步、真实大规格 golden 和公开性能剖面。也应保留其范围：对非整除形状、极小 K、非方阵和 A2 不保证沿用当前参数仍最优。本机仅检查文件和脚本，未运行 NPU 编译与测量。[复现命令](reproduce.md)。
