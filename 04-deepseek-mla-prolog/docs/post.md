# DeepSeek MLA 前处理融合：A2/A3 上的 Split-N、反量化流水和 KV Cache 写入

`MlaPrologV3` 把 MLA 的多次矩阵乘、RMSNorm、RoPE、量化和 Cache 更新放进同一个混合 Cube/Vector 内核。本案例固定 **CANN 9.0.0 正式标签 `v9.0.0`** 的实现，只研究 **Atlas A2/A3 的 BF16 / INT8 路径**，逐函数说明它怎样处理前后依赖。

它属于单个融合算子的内部优化；本文不使用模型 TPOT 或 tokens/s 作为性能证据。未找到这个固定版本的公开 A/B 单算子耗时，本机也没有 NPU，因此以下结论分为可核对的源码事实与需要 profiler 验证的收益解释。[复现步骤](reproduce.md) 和 [来源证据](sources.md) 将两者分开记录。

## 1. 为什么不能只保存 V3 的入口文件

固定提交是：

```text
ops-transformer v9.0.0
afe72144f9f2ac8441929035795db88a111b30c5
```

[CANN 9.0.0 发布说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) 明确配套这个正式标签，且列出旧 `MlaProlog/MlaPrologV2WeightNz` 接口迁移到 V3 的方向。本文不依据名称模糊的 `cann9.0` 镜像判断版本。

V3 的 [kernel 入口](../code/upstream/attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp) 只是定义 `MLA_PROLOG_VERSION=3`，实例化模板并调用 `Init/Process`。真正的主流水在：

```text
attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h
```

V3 的 [host CMake](../code/upstream/attention/mla_prolog_v3/op_host/CMakeLists.txt) 明确声明依赖 `attention/mla_prolog`；[tiling 注册](../code/upstream/attention/mla_prolog_v3/op_host/mla_prolog_v3_tiling_register.cpp) 也调用共享 `TilingMlaProlog`。因此本 post 一并保存了共享 Kernel、Matmul/RMSNorm/RoPE/Scatter 服务和共享 Host Tiling，避免得到“只有 API 包装器、没有优化实现”的资料。

硬件证据有两层：[产品支持表](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/mla_prolog_v3/README.md) 列出 A2/A3；[V3 注册文件](../code/upstream/attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp) 第 171–172 行实际注册 `ascend910b` 与 `ascend910_93`。同文件给 950 建立独立配置。共享 RMSNorm 服务用编译条件选择 A2/A3 的 `arch32/rms_norm.h`，另一侧的 `arch35/vf/` 不在本文研究范围。

## 2. 真实计算图：四路结果，四类矩阵乘

输入 `x` 同时产生 Query 与低秩 KV。Query 路径先做低秩投影与归一化，再计算 non-RoPE Query 和 RoPE Query；KV 路径生成低秩 Cache 与位置编码 Cache：

\[
c^Q=\alpha_q\operatorname{RMSNorm}(xW^{DQ}),\quad
q^C=c^QW^{UQ},\quad q^N=q^CW^{UK},\quad
q^R=\operatorname{RoPE}(c^QW^{QR})
\]

\[
c^{KV}=\alpha_{kv}\operatorname{RMSNorm}(xW^{DKV}),\quad
k^C=\operatorname{Cache}(c^{KV}),\quad
k^R=\operatorname{Cache}(\operatorname{RoPE}(xW^{KR}))
\]

上游把 `W_UQ` 与 `W_QR`、`W_DKV` 与 `W_KR` 分别拼在一起，减少重复读输入并把两个输出合进一次矩阵乘。这个融合算子内部的四类 Matmul 为：

| 内核阶段 | 左/右矩阵含义 | 后续消费者 |
| --- | --- | --- |
| `MatmulCq` | x × W_DQ | RMSNorm Cq |
| `MatmulCkvKr` | x × 拼接的 W_DKV_KR | RMSNorm Ckv、RoPE Kr、Cache |
| `MatmulQcQr` | cQ × 拼接的 W_UQ_QR | Qc 反量化 / Qr RoPE |
| `MatmulQn` | 分 head 的 Qc × W_UK | non-RoPE Query，必要时动态量化 |

输入 token 的 B/S 轴合为 M，主循环每次处理至多 128 个 token。投影权重主要为 FRACTAL_NZ，`W_UK` 为 ND；`service_matmul.h` 在搬运中把 ND 转换为 Cube 可用的片上布局。把所有权重都称为 NZ 或把 NZ 误写成一个 view，会导致复现错误。

## 3. 先按输出列切 Matmul，再按完整行切 RMSNorm

小 batch 的 MLA 投影常有很宽的输出列，沿 N 切可让多核参与单个或少量 token 的 Matmul。共享 [Host Tiling](../code/upstream/attention/mla_prolog/op_host/mla_prolog_tiling.cpp) 中的 `FillMatmul1Tiling、FillMatmul2Tiling、FillMatmul3Tiling、FillMatmul4Tiling` 分别计算各阶段的列/head 分核信息。

RMSNorm 需要一整行的平方和，不能对某个 Cube 的 N 片段独立归一化。因而 Matmul 按列分核后必须先汇齐；Vector 按 token 行处理 RMSNorm，再在下一次矩阵乘开始前确保这些行都写完。

在 [`AicProcess/AivProcess`](../code/upstream/attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h) 中可直接核对依赖链：

| 生产者 | 同步标志 / 边界 | 消费者 |
| --- | --- | --- |
| Cube MatmulCq 的 Fixpipe | `FINISH_MM_CQ`，随后等待全核相关边界 | Vector RMSNorm Cq |
| Vector RMSNorm Cq 的 MTE3 | `FINISH_VEC_RMSNORM_CQ` | Cube MatmulQcQr |
| Cube MatmulCkvKr 的 Fixpipe | `FINISH_MM_CKVKR` | Vector RMSNorm / RoPE / Scatter |
| Vector 完成本批 Cache 更新 | `FINISH_VEC_CKVKR` | 下一轮 CkvKr 生产与缓冲复用 |
| Cube MatmulQcQr | 整体或 Split-N 的 Qc/Qr 就绪标志 | Vector 反量化 / RoPE，后续 Cube Qn |

其中 `WaitAllCore` 与 CrossCore flag 用途不同：前者保护跨核切分重新组合的完整数据，后者保护 Cube/Vector 之间的生产消费。融合不意味着能随意删除同步。

## 4. 反量化不必等整个 QcQr：按 head 子块推进

`ProcessBaseInputs` 会为适合的 INT8 shape 置 `enableDequantOpt_`；除特殊算力分组路径外，当 `weightUqQr` 为 INT8 且 Query head 数达到相应条件，就采用能保留 head 边界的 N 切分。

`MatmulQcQr` 每生产一个 N 片段就设置 `FINISH_MM_QCQR_SPLIT_N`。Vector 的 `DequantAndRopeSplitNSyncMMQcQr` 等待片段就绪，确认当前片段已经覆盖某个 head 的 Qc 列后，才执行 `DequantQcQrSplitN`；齐备的 Qr 片段则交给 `RopeQrSplitN`。

反量化后的 Qc 写回后设置 `FINISH_VEC_DEQUANT_QC_SPLIT_N`。后续 `MatmulQnSyncDynamicQuantAndMulQr` 能按 head 读取已完成的 Qc，形成：

```text
Cube:   QcQr 的 N 子块生产 → 下一子块生产 → Qn 按 head 消费
Vector:      Qc 反量化 / Qr RoPE → 下一 head 的反量化 / RoPE
```

这比“QcQr 全部结束、Vector 全部反量化结束、Qn 再开始”更有机会重叠。它的成本是每个子块的同步和更复杂的布局偏移，所以 Host 必须保证切分不把 head 内的 Qc/Qr 关系切坏。

还有一个容易漏掉的限制：`MAX_SYNC_FLAG_COUNT=15`。当 head 子循环数量较大时，源码使用每两个 head 等待/通知一次并照顾最后一个 head，避免同一 flag 计数累积超过硬件允许的范围。增加 head 数而没有同步计数保护，并不是安全的“只加循环”修改。

## 5. 两个明确的访存优化：A 全载和第一份 Qn 权重预取

### A 全载的门槛来自 L1 容量

`mla_prolog_comm.h` 明确给出 `INT8_AFULLLOAD_MAX_MSIZE=64`、`BF16_AFULLLOAD_MAX_MSIZE=32`。`MatmulQcQr` 在 M 不超过对应门槛时，先用 `LoadL1A` 载入整个 A，再循环 N 子块复用这份 A；超过门槛则回到按块搬运。

这处理的是宽 N 矩阵乘重复加载左矩阵的问题。BF16 元素宽度是 INT8 的两倍，所以两种数据类型的 M 门槛不同。不能把 M=128 的主循环上限直接当成所有场景都可 A 全载。

L1 A/B、L0 A/B/C 在共享 Matmul 服务中使用 ping-pong 缓冲与 HardEvent 控制。源码保留 `MTE1_MTE2`、`M_MTE1`、`FIX_M` 等标志，分别保护搬运、Cube 计算和输出之间的局部缓冲复用。

### Qn 权重预取先覆盖第一个 head

`PreloadQnAndSync → MatmulQnWeightPreload` 在等待 Qc 数据之前，预先加载第一份 `W_UK` 到 L1 B，随后第一次 `MatmulFullLoad<...,true,true>` 使用它。后续 head 走相应的常规加载分支。

这里应准确理解为“把第一份 Qn 权重搬运放到依赖等待前”，不能夸大成“所有 head 权重永久驻留 L1”。只有实际 profiler 的 MTE2 时间与等待区间，才能证明具体 shape 上隐藏了多少搬运时间。

## 6. 小到只有一个 token 时，算力分组不是普遍开关

共享 Host 的 `ProcessBaseInputs` 只在特殊条件下开启 `enableGroupComputeOpt_`：半量化模式、token 总数为 1、`nkvSize=8`、至少 16 个 AIC / 32 个 AIV，且不是 CV 1:1。进入后限定使用 16 AIC / 32 AIV。

在 `MatmulAndSyncQcQr` 内，Qc 与 Qr 被分给不同的核集合，常量 `QC_CORE_NUM=8`、`QR_CORE_NUM=4`。Qc 结果就绪后可以先推进对应反量化和 Qn，而不用把两条输出支路完全绑成一次统一等待。

这是一种针对小 M、不同分支计算量不对称的专门实现。它说明高质量优化往往是**特定 shape 的模板选择**，而非“一律核数越多越好”。复现时至少比较 token=1 与更大 batch、半量化与非量化；仅调用 BF16 的官方示例不会自动覆盖这个路径。

## 7. RMSNorm、量化与 Cache 写入融合在 Vector 一侧

`CopyGlobalParams` 在首轮加载 Cq/Ckv gamma、反量化权重 scale、smooth scale 和适用的 Cache 量化 scale，后续 token 复用它们。`CopyInSinCos` 准备 RoPE 数据。

`RmsNormCq` 将投影结果读入 UB，按需要反量化、计算 RMSNorm、smooth 与动态量化，再写入下一次 Matmul 的输入 workspace。A2/A3 的 [`service_rms_norm.h`](../code/upstream/attention/mla_prolog/op_kernel/service_rms_norm.h) 调用 [`arch32/rms_norm.h`](../code/upstream/attention/mla_prolog/op_kernel/arch32/rms_norm.h) 和非 VF 的动态量化实现；不会执行 950 的 VF 分支。

Cache 支路由 `RmsNormRopeScatterCkvKr` 组织，内部调用 `RmsNormAndScatterCkv` 与 `RopeAndScatterKr`。计算完的 Ckv/Kr 在同一 Vector 内核中直接写到指定 Cache 位置，不再单独提交一个 Scatter 算子。PA 模式用 `cacheIndex`；块模式通过 `ComputeBlkScatterOffsets` 和 `ScatterCacheMultiRows` 处理跨页与实际序列长度；NZ/BSND 模式的物理偏移不同，必须按 Cache 契约生成张量。

融合后仍保留若干 GM 中间结果，比如 Cq/CkvKr Matmul 输出、RMSNorm Cq 输出、Qc 反量化结果。片上复用降低了部分参数读取和独立算子开销，但这份实现并不承诺“全流程无 HBM 中间数据”。Workspace 的布局和跨核依赖正是使用 GM 的原因之一。

Cache INT8 会减少后续 Cache 数据量，却增加当前算子的量化操作、scale 存储和精度要求。per-tensor、per-channel、per-tile 的布局与 scale 处理不同，本文主要学习 BF16/INT8 常规路径；新增模式必须对照 API 支持组合，不能因为入口模板出现某个枚举就认定所有硬件和组合都能运行。

## 8. 复现时应怎样证明优化有效

官方 [aclnn 样例](../code/upstream/attention/mla_prolog_v3/examples/test_aclnn_mla_prolog_v3.cpp) 使用 B=8、S=1、He=7168、Hcq=1536、128 个 head、D=128、Hckv=512、Dr=64。权重和输入初始化接近全零，它适合作为接口 smoke test，不能替代随机数值验证或性能样例；随机 NZ 权重需要做真实布局转换，不能仅修改维度。

建议先用相同 dtype 和 Cache 模式拆开的 Matmul/RMSNorm/RoPE/Cache 流程形成参照，检查 Query、Query Rope、Cache 更新区域与 scale。再只测量 device 计算区间，排除 host 初始化、权重转换和申请内存时间，记录单融合算子耗时与拆分链的总耗时。

针对本源码的三项优化，实验应分别覆盖：M 在 INT8 64/BF16 32 门槛两侧；token=1 且满足/不满足算力分组条件；head 数足以观察 Split-N 子块消费的场景。记录 kernel tiling、flag 等待、MTE2/FIX 与 Vector 时间，确认收益是否来自参数复用、搬运重叠或减少提交开销。未取得这些数据前，本文只认定实现机制存在，不给出定量加速承诺。

本案例最值得借鉴的是**在算子内部识别两条可独立推进的输出支路，并按消费者的就绪粒度重新切分和同步**。完整源码阅读入口及构建步骤见 [原始源码](../code/upstream/) 与 [reproduce.md](reproduce.md)。
