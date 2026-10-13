# CANN Ascend C 算子优化案例

面向 **CANN 9.0.0 及以上版本、Ascend A2（910B）与 A3** 的 Ascend C 算子源码与实战资料集合。重点收录算子内部的分块、数据搬运、Cube–Vector 流水、同步、量化和计算融合实现，并优先保留有原始优化记录、基线与算子性能对比的案例。

当前共 **67 个案例**，包含 **40 个 CANN 官方算子库案例**与 **27 个比赛、社区、SGLang 和研究项目案例**。设备支持与 CANN 版本按案例区分，详见下方支持状态。

## 从有性能对比的案例开始

以下数字引用对应的上游原始报告，测量环境、输入形状和计时范围随报告保留。

| 案例 | 可以重点阅读的实现 | 原始报告中的性能依据 |
| --- | --- | --- |
| [05 · Causal Conv1d Cast-once](./05-community-causal-conv1d-cast-once-a2/) | FP32 状态环只在边界转换，均衡 6144 通道的分块，减少循环内 Cast | [PR 原文](./05-community-causal-conv1d-cast-once-a2/docs/PR-10507.md)：910B2，48 组配置的设备耗时中位数之和 53,427.6 → 42,704.5 μs，比值 1.251×；包含完整逐形状表 |
| [07 · GDN QKV 与 Gating 融合](./07-community-gdn-qkv-gating-fusion-a2-a3/) | 将 QKV 重排和 GDN gating 合并，减少搬运、临时张量与 launch | [PR 原文](./07-community-gdn-qkv-gating-fusion-a2-a3/docs/PR-16895.md)：8192 tokens 的算子 Graph Replay 耗时 253.158 → 64.562 μs，3.921×；Eager 与 Graph 两套口径分别列出 |
| [08 · GMM NZ Scale Sync](./08-community-gmm-nz-scale-sync-a2/) | 用 `PIPE_V`、`V_S`、`S_V` 表达实际依赖，缩小 `PIPE_ALL` 同步范围 | [PR 原文](./08-community-gmm-nz-scale-sync-a2/docs/PR-15725.md)：910B2 / CANN 9.0.1，tiling key 1 设备平均耗时 373.094 → 364.210 μs；保留对照路径、配对样本与置信区间 |
| [12 · BatchMatmulMaxSum](./12-contest-batchmatmul-maxsum-a3/) | 形状驱动的分块、多核归约、Cube–Vector 流水，以及退化方案的撤回记录 | [优化记录](./12-contest-batchmatmul-maxsum-a3/docs/optimal_log.md)：910C / CANN 9.0.0，宽 N 分块调整后 OJ case 8 从 92.88 降至 85.11 μs；同文分别记录 OJ 与远端 `msprof` 测量 |

性能数值只适用于报告中的环境和工作负载。设备 Kernel 时间、含 Host 调度的单算子调用时间、Graph Replay 时间分别比较；上游材料中的模型吞吐或服务延迟保留其原始口径。各案例的资料丰富程度不同，CANN 官方案例主要提供内核实现与接口约束，比赛和社区优化记录通常包含具体实验过程。

## 收录范围

`code/` 保留以下内容：

- Ascend C 算子内核和直接相关的计算模块。
- Tiling、算子注册、形状推导及直接 Host/API 适配。
- A2/A3 路径实际引用的最小共享源码与声明。

`docs/` 保存上游原始算子文档、作者优化记录、相关项目说明和 PR 原文。算子融合可以包含多个计算阶段，例如 Dispatch＋FFN＋Combine、Matmul＋ReduceScatter；收录范围落在这些算子的内核和直接接口。

本仓排除 Triton、PTO、950/310P 专用代码，以及模型运行器、服务端、项目级 Buffer/分配器、训练与推理流程。每个案例中不放构建、复现、测试、校验脚本、源码清单或独立许可证文件；源码自身的版权注释保留。

明确不支持 A2 的案例不收录；只有更新版 CANN 依据、同时缺少明确 A2 支持或验证依据的案例也不收录。已有 CANN 9.0.0＋A3 记录以及 A2 工程适配的保留案例，分别标注其待补证据。

源码采用固定上游提交，并按目标范围整理目录、修正 include 路径、裁剪不相关分支与依赖。具体来源与提交可结合本仓 Git 历史和案例内的原始资料追溯。

## 目录结构

```text
.
├── README.md
├── 01-add-layernorm/
│   ├── code/
│   │   ├── common/
│   │   ├── op_graph/
│   │   ├── op_host/
│   │   └── op_kernel/
│   └── docs/
├── …
└── 67-weight-quant-batch-matmul/
    ├── code/
    │   ├── common/
    │   ├── op_host/
    │   └── op_kernel/
    └── docs/
```

每个案例的 `code/` 与 `docs/` 并列。根据上游实现，代码也可能使用 `kernels/`、`tiling/`、`op_api/`、`op_graph/`、`utils/`、`include/` 或关联算子目录；单文件方案直接保存为 `code/kernel.asc`。编号按去掉数字前缀后的完整英文文件夹名排序，连续从 `01` 到 `67`。

阅读内核时，可先看 Host 的分派和 Tiling，再定位 Kernel 入口及其计算、搬运、同步路径。`lib/matmul_intf.h`、`kernel_operator.h` 等由 CANN SDK 提供；Torch/TorchNPU 接口和上游生成的 launch 头也属于外部配套依赖。完整构建环境与工程入口由对应上游项目提供。

## 版本与 A2 支持状态

这里的 **A2 指 Ascend 910B 系列**。**CANN ≥9.0.0** 与 **CANN 9.0.0 本身**的兼容证据分别记录，工程配置、产品支持声明和真机报告也分别看待。

所有 67 个案例的核心内核均为 Ascend C。下表描述现有上游证据，不代表本仓逐例完成了 NPU 运行验证。

| 标记 | 含义 | 数量 | 案例编号 |
| --- | --- | ---: | --- |
| **S** | 有明确的 CANN 9.0.0 配套与 A2 支持依据：正式版本支持表、项目声明或原始验证记录 | 51 | 01–05、13–14、16–17、20–52、59–67 |
| **E** | 有 CANN 9.0.0/9.1.0 工程与 A2 接入依据，缺少 9.0.0＋A2 的单算子组合验证 | 8 | 18–19、53–58 |
| **V** | 有明确 A2 支持或测试说明，当前版本证据是 9.0.1/9.1.0，9.0.0 本身待确认 | 5 | 06–09、11 |
| **A3** | 已有 CANN 9.0.0＋A3 测试记录，A2 支持待补依据 | 3 | 10、12、15 |

E 类来自 SGLang 原生算子：其 [CMake 配置](https://github.com/sgl-project/sgl-kernel-npu/blob/00db6cd6bd2280650281ea187292fe8913619abf/csrc/CMakeLists.txt)允许 A2/A3 架构族，`SGL_KERNEL_ENABLE_A3_ONLY_OPS` 开关在该版本也覆盖 A2；[算子测试 CI](https://github.com/sgl-project/sgl-kernel-npu/blob/00db6cd6bd2280650281ea187292fe8913619abf/.github/workflows/pr-test-kernels.yml)的设备矩阵只列 A3。DeepEP 的 A2 CI 覆盖不能直接套用到其他算子。

V 类的具体版本依据为：08、11 使用 9.0.1；09 使用 9.1.0；06、07 的配套源码使用 9.1.0。它们符合 CANN ≥9.0.0 的版本范围。各算子的 dtype、shape、布局和通信拓扑限制仍以对应原始文档为准。

## 案例目录

### Attention、位置编码与 KV Cache · 14 个

| 编号 | 案例 | 来源 | 支持状态 |
| --- | --- | --- | --- |
| 16 | [DeepSeek Lightning Indexer](./16-deepseek-lightning-indexer/) | CANN | S |
| 17 | [DeepSeek MLA Prolog](./17-deepseek-mla-prolog/) | CANN | S |
| 18 | [DeepSeek SGL MLA Preprocess](./18-deepseek-sgl-mla-preprocess/) | SGLang | E |
| 19 | [DeepSeek SGL Sparse Shared-KV Attention](./19-deepseek-sgl-sparse-shared-kv-attention/) | SGLang | E |
| 20 | [DeepSeek Sparse Flash Attention](./20-deepseek-sparse-flash-attention/) | CANN | S |
| 22 | [Dequant RoPE Quant KVCache](./22-dequant-rope-quant-kvcache/) | CANN | S |
| 26 | [Flash Attention Grad](./26-flash-attention-grad/) | CANN | S |
| 27 | [Flash Attention Score](./27-flash-attention-score/) | CANN | S |
| 28 | [Fused Floyd Attention](./28-fused-floyd-attention/) | CANN | S |
| 32 | [Incre Flash Attention](./32-incre-flash-attention/) | CANN | S |
| 45 | [NSA Compress](./45-nsa-compress/) | CANN | S |
| 46 | [NSA Selected Attention](./46-nsa-selected-attention/) | CANN | S |
| 47 | [QKV RMSNorm RoPE Cache](./47-qkv-rmsnorm-rope-cache/) | CANN | S |
| 51 | [RoPE](./51-rope/) | CANN | S |

### Matmul、FFN、量化与激活 · 12 个

| 编号 | 案例 | 来源 | 支持状态 |
| --- | --- | --- | --- |
| 08 | [GMM NZ Scale Sync](./08-community-gmm-nz-scale-sync-a2/) | vLLM Ascend 社区 PR | V |
| 23 | [Dequant SwiGLU Quant](./23-dequant-swiglu-quant/) | CANN | S |
| 24 | [Dynamic Quant](./24-dynamic-quant/) | CANN | S |
| 25 | [FFN](./25-ffn/) | CANN | S |
| 29 | [Grouped Matmul](./29-grouped-matmul/) | CANN | S |
| 34 | [Matmul](./34-matmul/) | CANN | S |
| 40 | [MoE Grouped Matmul SwiGLU Quant](./40-moe-grouped-matmul-swiglu-quant/) | CANN | S |
| 48 | [Quant Batch Matmul](./48-quant-batch-matmul/) | CANN | S |
| 56 | [SGL SoftFP8 Grouped Matmul](./56-sgl-softfp8-grouped-matmul/) | SGLang | E |
| 57 | [SGL SoftFP8 Matmul](./57-sgl-softfp8-matmul/) | SGLang | E |
| 59 | [SwiGLU](./59-swiglu/) | CANN | S |
| 67 | [Weight Quant Batch Matmul](./67-weight-quant-batch-matmul/) | CANN | S |

### 归一化与 Softmax · 8 个

| 编号 | 案例 | 来源 | 支持状态 |
| --- | --- | --- | --- |
| 01 | [Add LayerNorm](./01-add-layernorm/) | CANN | S |
| 02 | [Add RMSNorm Dynamic Quant](./02-add-rmsnorm-dynamic-quant/) | CANN | S |
| 03 | [Add RMSNorm Quant](./03-add-rmsnorm-quant/) | CANN | S |
| 31 | [GroupNorm SiLU](./31-groupnorm-silu/) | CANN | S |
| 33 | [LayerNorm](./33-layernorm/) | CANN | S |
| 49 | [RMSNorm](./49-rmsnorm/) | CANN | S |
| 50 | [RMSNorm Grad](./50-rmsnorm-grad/) | CANN | S |
| 52 | [Scaled Masked Softmax](./52-scaled-masked-softmax/) | CANN | S |

### MoE 路由与通信计算融合 · 12 个

| 编号 | 案例 | 来源 | 支持状态 |
| --- | --- | --- | --- |
| 04 | [AllGather Matmul](./04-all-gather-matmul/) | CANN | S |
| 11 | [W8A8 Fused MoE](./11-community-w8a8-fused-moe-a2-pr-13426/) | vLLM Ascend 社区 PR | V |
| 30 | [Grouped Matmul Finalize Routing](./30-grouped-matmul-finalize-routing/) | CANN | S |
| 35 | [Matmul ReduceScatter](./35-matmul-reduce-scatter/) | CANN | S |
| 36 | [MoE Dispatch／Combine](./36-moe-dispatch-combine/) | SGLang DeepEP | S |
| 37 | [MoE Finalize Routing V2](./37-moe-finalize-routing-v2/) | CANN | S |
| 38 | [MoE Gating TopK](./38-moe-gating-topk/) | CANN | S |
| 39 | [MoE Gating TopK Softmax](./39-moe-gating-topk-softmax/) | CANN | S |
| 41 | [MoE Init Routing Quant](./41-moe-init-routing-quant/) | CANN | S |
| 42 | [MoE Init Routing V2](./42-moe-init-routing-v2/) | CANN | S |
| 43 | [MoE Token Permute](./43-moe-token-permute/) | CANN | S |
| 44 | [MoE Token Unpermute](./44-moe-token-unpermute/) | CANN | S |

### 卷积 · 3 个

| 编号 | 案例 | 来源 | 支持状态 |
| --- | --- | --- | --- |
| 05 | [Causal Conv1d Cast-once](./05-community-causal-conv1d-cast-once-a2/) | vLLM Ascend 社区 PR | S |
| 21 | [Deformable Conv2d](./21-deformable-conv2d/) | CANN | S |
| 53 | [SGL Causal Conv1d](./53-sgl-causal-conv1d/) | SGLang | E |

### LoRA、状态更新、mHC 与数据重排 · 6 个

| 编号 | 案例 | 来源 | 支持状态 |
| --- | --- | --- | --- |
| 06 | [GDN QKV DMA](./06-community-gdn-qkv-dma-a2-a3/) | vLLM Ascend 社区 PR | V |
| 07 | [GDN QKV Gating Fusion](./07-community-gdn-qkv-gating-fusion-a2-a3/) | vLLM Ascend 社区 PR | V |
| 09 | [mHC Expand Grouped Copy](./09-community-mhc-expand-grouped-copy-a2/) | vLLM Ascend 社区 PR | V |
| 54 | [SGL LoRA Cube／Vector](./54-sgl-lora-cube-vector/) | SGLang | E |
| 55 | [SGL Recurrent Gated Delta Rule](./55-sgl-recurrent-gated-delta-rule/) | SGLang | E |
| 58 | [SGL Sparse KV Copy／Lookup](./58-sgl-sparse-kv-copy-lookup/) | SGLang | E |

### 比赛与独立社区方案 · 5 个

| 编号 | 案例 | 来源 | 支持状态 |
| --- | --- | --- | --- |
| 10 | [RoPE RowGroup](./10-community-rope-rowgroup-a3/) | 社区方案 · dy_dessert | A3 |
| 12 | [BatchMatmulMaxSum](./12-contest-batchmatmul-maxsum-a3/) | 比赛方案 · liang_yue | A3 |
| 13 | [mHC Expand](./13-contest-mhc-expand-a2/) | 比赛方案 · marak999459 | S |
| 14 | [mHC Sinkhorn](./14-contest-mhc-sinkhorn-a2/) | 比赛方案 · marak999459 | S |
| 15 | [QuantMatmul ReLU Quant](./15-contest-quantmatmul-relu-quant-a3/) | 比赛方案 · liu-jingzong | A3 |

### TCUSCAN 并行算法与研究实现 · 7 个

| 编号 | 案例 | 来源 | 支持状态 |
| --- | --- | --- | --- |
| 60 | [Prefix Scan](./60-tcuscan-prefix-scan/) | Huawei CSL | S |
| 61 | [Radix Sort](./61-tcuscan-radix-sort/) | Huawei CSL | S |
| 62 | [Segmented Scan／Sum](./62-tcuscan-segmented-scan-sum/) | Huawei CSL | S |
| 63 | [Sparse Matrix–Vector](./63-tcuscan-sparse-matrix-vector/) | Huawei CSL | S |
| 64 | [Stream Compaction](./64-tcuscan-stream-compaction/) | Huawei CSL | S |
| 65 | [TopK](./65-tcuscan-topk/) | Huawei CSL | S |
| 66 | [Triangular Inverse](./66-tcuscan-triangular-inverse/) | Huawei CSL | S |

## 上游来源

| 来源 | 案例数 | 上游项目 |
| --- | ---: | --- |
| CANN 官方算子库 | 40 | [ops-transformer](https://gitcode.com/cann/ops-transformer)、[ops-nn](https://gitcode.com/cann/ops-nn)，取自正式 v9.0.0 |
| SGLang 原生算子与 DeepEP | 9 | [sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) |
| vLLM Ascend 社区优化 PR | 6 | [vllm-ascend](https://github.com/vllm-project/vllm-ascend)，具体作者分支与 PR 见案例资料和对应提交说明 |
| 比赛与独立社区仓库 | 5 | [acend-race](https://gitcode.com/liang_yue/acend-race)、[cann-1](https://gitcode.com/liu-jingzong/cann-1)、[rotaryposemb](https://gitcode.com/dy_dessert/rotaryposemb)、[ascend-operator-competition](https://github.com/marak999459/ascend-operator-competition) |
| Huawei CSL 研究项目 | 7 | [pytorch-tcuscan](https://github.com/huawei-csl/pytorch-tcuscan) |

DeepSeek 相关案例来自上述不同项目，包括 LI、SFA、MLA、MoE、NSA 和 KV Cache 实现。社区 PR 方案按其固定源码版本和原始报告收录，收录本身不表示已经合入上游正式版本。TCUSCAN 单独归为研究项目来源。
