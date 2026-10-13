# 案例纳入标准与排除记录

纳入案例需同时具备：CANN 9.0.0 或更高版本的明确配套/兼容依据；A2 910B 或 A3 910_93 的产品与实际内核路径；使用 Ascend C 编写的设备内核；可以定位到 Tiling、缓冲、指令或数据流的具体实现。各 post 对应独立内核或算子组合，附源码、来源和复现步骤。Host、构建配置和 Python 测量脚本属于必要配套文件。

正式版本以[发布管理仓库的 9.0.0 配套表](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)为准。`9.0.0` 分支可能继续移动，本合集使用正式 `v9.0.0` tag 的固定 commit。后来新增但明确兼容 CANN 9.0 的案例，在目录和文章中单独说明。

## 本次排除的候选

以下判断针对核查过的固定版本与目录，不能扩展为所有版本或所有同名 API 的限制。

| 候选 | 排除依据 | 采用的相关案例 |
|---|---|---|
| ops-nn `softmax_v2` | 本次 v9.0.0 快照为 arch35 / 950 路径；不能因目录名是 Softmax 就认作 A2/A3 实战。[Host 注册](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/activation/softmax_v2/op_host/softmax_v2_def.cpp)。 | [02 的在线 Softmax](02-deepseek-sparse-flash-attention/docs/post.md) |
| ops-nn `layer_norm_v3` | 本次 v9.0.0 的目标注册与实现为 950 路径。[Host 注册](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/layer_norm_v3/op_host/layer_norm_v3_def.cpp)。 | [10 LayerNormV4](10-layernorm/docs/post.md) |
| ops-math `top_k_v2` | v9.0.0 Host 注册 `ascend950` / `mc62cm12a`，缺少目标 A2/A3 注册。[Host 注册](https://gitcode.com/cann/ops-math/blob/ce515930d875e6fb0187195e0b328ab2def4bea5/math/top_k_v2/op_host/top_k_v2_def.cpp)。 | [01 内部的 TopK 归并](01-deepseek-lightning-indexer/docs/post.md) |
| ops-math `reduce_sum` | 同一固定快照 Host 注册为 `ascend950` / `mc62cm12a`。[Host 注册](https://gitcode.com/cann/ops-math/blob/ce515930d875e6fb0187195e0b328ab2def4bea5/math/reduce_sum/op_host/reduce_sum_def.cpp)。 | [08 RmsNorm 归约](08-rmsnorm/docs/post.md) |
| ops-nn `conv2d_v2` / `extend_conv2d` | 检查到的常规 Host 与 arch35 实现不能证明所要求的 A2/A3 内核路径。[Conv2DV2 Host](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/conv2d_v2/op_host/conv2d_v2_def.cpp)、[ExtendConv2D Host](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/extend_conv2d/op_host/extend_conv2d_def.cpp)。 | [22 DeformableConv2d](22-deformable-conv2d/docs/post.md) |
| 仅提供 Conv API 封装的 A2 示例 | 找不到可解析的真实卷积内核，不能用 API 调用示例代替算子内部优化。 | [22](22-deformable-conv2d/docs/post.md) |
| 950 专属 DeepSeek GEMM、MXFP4 或 EP 示例 | 设备或指令路径超出 A2/A3；不同仓库中同名模型不能作为硬件证据。 | 01–05、13–15 |
| 模型吞吐、框架调度、整机部署文章 | 缺少具体算子内核、shape 与实现；整模型吞吐不能充当算子优化成绩。 | 各 post 的内核分析和测量步骤 |

## 三个需单独保留的硬件限制

- [02 SparseFlashAttention](02-deepseek-sparse-flash-attention/docs/sources.md)：README 列 A2/A3 推理，但固定 Host 仅接受 `ASCEND910B`。复现范围收紧到 A2 推理；A3 支持未按这份源码建立。
- [21 FusedFloydAttention](21-fused-floyd-attention/docs/sources.md)：A2 支持训练产品，A3 支持训练/推理产品。A2 推理未列入。
- [23 FlashAttentionScoreGrad](23-flash-attention-grad/docs/sources.md)：A2/A3 训练产品。版本发布表的 SAMEAB/BN2 平均优化没有逐 shape、逐硬件明细。

## 性能数据的使用规则

21 保留公开算子 A/B 数据，并注明历史测量 CANN/驱动信息缺失；05 保留绝对参考时延；23 保留官方版本级模板汇总。其余案例以生产源码证明机制，列出后续消融与 profiling 步骤，不填造速度提升。

“使用 CANN 9.0.0 配套源码”证明源码的配套关系；“历史时延在 9.0.0 发布材料中”不能自动证明其测试环境是 CANN 9.0.0。每篇 `metadata.json` 分开记录这两类证据。
