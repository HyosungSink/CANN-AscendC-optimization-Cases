# DeepSeek 风格 MoE 的 Dispatch / Combine：把路由、量化和通信做到算子内部

这篇拆解的是 **SGLang `sgl-kernel-npu` 中的 DeepEP 通信算子**。对象是 `MoeLowLatencyDispatchV2` 与 `MoeLowLatencyCombineV2`，不是整网吞吐调优，也不是 `deepseek-ai/DeepEP-Ascend` 的 Ascend 950 内核。

固定源码 commit：`653e519cf9556b0cc8d9e7966aa4551eb65143ae`。上游 README 和 CI 配置明确覆盖 **Atlas A2 / A3，CANN 9.0.0 / 9.1.0**。本地保留了原始 Kernel、Host Tiling、调用桥接和测试代码。这里只解释 A2/A3 路径；同一上游文件中的 A5/950 分支不在文章范围内。[版本和硬件证据](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/python/deep_ep/README.md)、[固定 CI 配置](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/.github/workflows/pr-test-deepep-npu.yml)。

## 1. 要优化的不是普通 AllToAll

输入为 `x[num_tokens, hidden]`、`expert_ids[num_tokens, topk]`。每个 token 可以去多个专家；专家又分布在多个 rank。Dispatch 必须同时完成：

1. 根据 expert id 计算目的 rank 和本地专家编号。
2. 在接收端按专家组织 token，并输出每个专家的有效数量。
3. 保存源 `rank_id / token_id / topk_id`，使 Combine 能返回原位置。
4. 选择性地把 BF16 token 动态量化为 INT8，并携带 scale。

如果把这些步骤拆成多个独立算子，再通过通用集合通信发送，额外的中间读写和启动会落在每层 MoE 的关键路径上。本案例在 AIV 内完成路由、量化、远端窗口写入和结果整理；Combine 再融合返回通信与 top-k 加权归约。

## 2. A3 Dispatch：先算目标地址，再写远端窗口

阅读 [`moe_distribute_dispatch_v2.h`](../code/upstream/csrc/deepep/ops/op_kernel/moe_distribute_dispatch_v2.h) 中 `SendToMoeExpert`、`QuantProcess`、`SetStatus` 和 `Process`。

`SendToMoeExpert` 把有效的 token-expert 配对分给 AIV。一个配对的目标 rank 是：

```text
to_rank = expert_id / experts_per_rank + shared_expert_rank_num
```

目标窗口进一步按“源 rank × 目的本地专家 × token 在该专家中的序号”定位。Token 经 `DataCopyPad` 搬到 UB；选择量化时调用 `QuantProcess`；`FillTriple` 填上反向路由三元组；随后将整个记录写到目的 rank 的通信窗口。这不是在 Python 层先做一个全局 permute 后才通信。

动态量化分支有具体实现：BF16 转 FP32、绝对值与最大值归约、乘 `127 / max_abs`、舍入转 INT8，并把反量化 scale `max_abs / 127` 写进记录。发送 payload 的主体从 `2 × hidden` 字节变为 `hidden` 字节，仍要支付 scale、路由信息、对齐与量化计算成本。**减少主体字节数不能直接等同于延迟减半**。

一个容易误读的 API 是 `use_fp8=True`：在这里它在 **A2/A3 选择 INT8**，不是 FP8 算术。它只是沿用 DeepEP 接口命名；A5 才有另外的 FP8/MX 模式。分析必须沿 `low_latency_strategy.py → deep_ep.cpp → quant_mode / tiling key` 跟到实际分支。[接口说明](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/python/deep_ep/doc/LOW_LATENCY_API.md)。

## 3. 完成通知与数据布局是算子设计的一部分

A3 的 `Process` 顺序是：

```text
AlltoAllDispatch → SetStatus → WaitDispatch → LocalWindowCopy
                                            → 可选 TP AllGather → UpdateTokenNumsOut
```

`SetStatus` 为各专家写发送计数和状态。在真正公布状态前，代码执行 `PipeBarrier<PIPE_ALL>()` 和跨核同步，确保此前的数据写入与状态发布有明确顺序。接收端等待状态，再由计数形成布局、搬入本地输出。计数不是用于事后打印的统计，它决定有效 token 的边界和输出组织方式。

这个实现也揭示了优化的代价：通信窗口容量、状态区、UB 队列和发送计数都成为接口约束。需要根据真实 `batch / EP / topk / hidden` 检查 `HCCL_BUFFSIZE`；过小的窗口不是一个可以靠重复计时掩盖的性能问题。具体容量公式见本地保存的 Low-Latency API 文档。

## 4. A3 Combine：按 token 等待，再融合加权求和

阅读 [`moe_distribute_combine_v2.h`](../code/upstream/csrc/deepep/ops/op_kernel/moe_distribute_combine_v2.h) 中 `ExpertAlltoAllDispatchCopyAdd`、`WaitDispatch`、`ProcessMoeExpert`、`LocalWindowCopy`。

专家输出按保存的路由信息写回源 rank 的窗口。`WaitDispatch(tokenIndex)` 观察当前 token 应收到的专家记录对应的状态；有效专家 mask 会改变目标计数。数据到齐后，`ProcessMoeExpert` 从窗口取一条专家结果，转 FP32、乘对应 top-k 权重并累加。`LocalWindowCopy` 把 token 分给不同 AIV，并在 token 的全部贡献完成后写输出。

这里的融合点是“返回通信 + 路由恢复 + top-k 加权 Reduce”，不是后接一个独立的框架 Reduce。状态等待也在设备端执行。需要注意，源码存在多种可选通信量化路径，本篇性能表对应 **BF16 Combine**；不要把可选 INT8 Combine 和公开表中的 BF16 配置混为一谈。

## 5. A2 与 A3 不能只换一个设备名称

A3 README 给出的测试拓扑是纯 HCCS；A2 的实现区分机内 HCCS 与机间 RDMA，另有分层通信路径。本地保存的 `ops2/op_kernel/moe_distribute_dispatch_v2_layered.h` 和 `moe_distribute_combine_a2_layered.h` 是具体入口，而不是拿 A3 延迟表推算 A2 延迟。

A2 构建使用 `deepep2`；A3 使用 `deepep`。固定 CI 的 A2 容器标签是 `910b`。A3 平台通常称为 910C；这里保留上游平台名称，避免把 A3 的全 HCCS 拓扑误写成 A2 的 910B 拓扑。截至固定快照，A2 多节点 CI 已停用；单节点 CI 覆盖与保留的手动多节点测试是两种不同证据。[A2 专属说明](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/python/deep_ep/doc/A2_DEEPEP.md)。

## 6. 有公开性能表，但不能改写成 CANN 9.0.0 的前后加速比

上游公布了 A3 384 SuperPOD 的算子级数据：128 tokens、hidden 7168、top-8、INT8 Dispatch、BF16 Combine。

| EP | Dispatch | Combine |
|---|---:|---:|
| 8 | 132 μs | 126 μs |
| 16 | 139 μs | 135 μs |
| 32 | 153 μs | 151 μs |

这些是上游性能数据，并非本机复测；表格未单列测量所用 CANN、驱动版本及当时 commit，且没有通用 AllToAll 基线。因此它证明上游公开过这个工作负载的低延迟结果，**不能证明固定 commit 在 CANN 9.0.0 上必然复现同一耗时，也不能推出优化倍数**。CANN 9.0.0 / 9.1.0 的适用证据来自当前固定源码的支持说明和 CI，而不是这张历史表。[性能表来源](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/README.md)。

## 7. 测量边界比一个微秒数字更重要

固定版本 `test_low_latency.py` 有两种口径：

- `bench(test_func)` 用 NPU Event 包住 Dispatch、测试所需的 `per_token_cast_back`、Combine；预热 50 次、测量 50 次，丢弃第一个计时样本，并在计时前用 256 MB 张量冲刷 L2。**这一项包含测试反量化，不是两个通信 Kernel 延迟之和。**
- `bench_kineto` 用 NPU Profiler 匹配 `MoeLowLatencyDispatchV2 / MoeLowLatencyCombineV2` 的设备事件，分别求平均 duration；测试外还有大 Matmul 和分布式同步，以降低 rank 启动不均的影响。它度量设备事件，不能与前一项混作同一口径。

脚本还按 token 实际选中的专家数估算通信字节数。其变量 `num_fp8_bytes` 仍保留历史 FP8 命名和 scale 估计；A2/A3 实际走 INT8 时，这个公式不能当作精确 HCCS 线上字节计数。阅读性能时优先比较相同拓扑、相同量化配置下的 **latency**；带宽要确认其分子是原始数据量、BF16 等价数据量，还是工具实际统计的传输量。

## 8. 这个案例如何用于自己的算子优化

先按 [复现说明](reproduce.md) 运行固定版本测试，保留各 rank 的 Dispatch/Combine profiler duration、精度结果和完整环境信息。再在同一个 A2 或 A3 拓扑比较 INT8 / BF16 Dispatch：前者减少传输，后者省去量化计算，胜负取决于 batch、hidden、EP 和链路。不要跨拓扑比较“同名 API”而漏掉内部实现不同。

这个案例值得学的是实现的整体约束：目的地址直接计算、量化在发送前融合、路由信息随 payload 保存、状态发布顺序、接收布局整理以及返回阶段的加权归约。每项都有可阅读的源代码和可检查的正确性测试。**本机未安装 NPU/CANN，因此没有本机编译、设备正确性或性能结论。**
