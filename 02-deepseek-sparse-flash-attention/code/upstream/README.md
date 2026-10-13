# ops-transformer

## 🔥Latest News

- [2026/02] 新支持算子[mhc_post](experimental/mhc/mhc_post)、[mhc_pre](experimental/mhc/mhc_pre)、[mhc_res](experimental/mhc/mhc_res)。
- [2026/01] 新支持算子[grouped_matmul<<<>>>调用示例](examples/fast_kernel_launch_example/csrc/grouped_matmul)，方便用户自定义使用。
- [2026/01] 新支持算子[fused_floyd_attention](attention/fused_floyd_attention)、[fused_floyd_attention_grad](attention/fused_floyd_attention_grad)、[matmul_allto_all](mc2/matmul_allto_all)。
- [2025/12] 新增[QuickStart](QUICKSTART.md)，指导新手零基础入门算子项目部署（支持Docker环境）、算子开发和贡献流程。
- [2025/12] 优化指南类文档，聚焦[算子开发指南](docs/zh/develop/aicore_develop_guide.md)，明确最小交付件和关键示例代码，针对[Ascend/samples](https://gitee.com/ascend/samples/tree/master)仓算子提供迁移本项目的指导。
- [2025/12] 支持transformer类onnx算子插件，包括[NPUFlashAttention](attention/flash_attention_score/framework)、[NPUMultiHeadAttention](common/src/framework)、[NPUMoeComputeExpertTokens](moe/moe_compute_expert_tokens/framework)等。
- [2025/12] 新支持算子[kv_rms_norm_rope_cache](posembedding/kv_rms_norm_rope_cache)、[attention_update](attention/attention_update)、[attention_worker_scheduler](attention/attention_worker_scheduler)、[gather_pa_kv_cache](attention/gather_pa_kv_cache)、[kv_quant_sparse_flash_attention](attention/kv_quant_sparse_flash_attention)、[lightning_indexer_grad](attention/lightning_indexer_grad)、[mla_preprocess](attention/mla_preprocess)、[mla_preprocess_v2](attention/mla_preprocess_v2)、[grouped_matmul_swiglu_quant_v2](gmm/grouped_matmul_swiglu_quant_v2)、[attention_to_ffn](mc2/attention_to_ffn)、[ffn_to_attention](mc2/ffn_to_attention)。
- [2025/12] 开源算子支持Ascend 950PR/Ascend 950DT/KirinX90，可以通过[CANN Simulator](docs/zh/debug/cann_sim.md)仿真工具开发调试。
- [2025/11] 新支持算子[kv_quant_sparse_flash_attention](attention/kv_quant_sparse_flash_attention)、[lightning_indexer](attention/lightning_indexer)、[quant_lightning_indexer](attention/quant_lightning_indexer)、[sparse_flash_attention](attention/sparse_flash_attention)。
- [2025/11] 新支持示例算子[rope_matrix](experimental/posembedding/rope_matrix)和[all_gather_add](examples/mc2/all_gather_add)。
- [2025/11] 新增算子开发工程模板[NpuOpsTransformerExt](experimental/npu_ops_transformer_ext)，无缝集成PyTorch张量操作，支持自动微分和GPU/NPU统一接口。
- [2025/10] 新增[experimental](experimental)目录，完善[贡献指南](CONTRIBUTING.md)，支持开发者调试并贡献自定义算子。
- [2025/09] ops-transformer项目首次上线，开源算子支持Atlas A2/A3系列产品。

## 🚀概述

ops-transformer是[CANN](https://hiascend.com/software/cann) （Compute Architecture for Neural Networks）算子库中提供transformer类大模型计算的进阶算子库，包括attention类、moe类等算子，算子库架构图如下：

<img src="docs/zh/figures/architecture.png" alt="架构图"  width="700px" height="320px">

## 📝版本配套

本项目源码会跟随CANN软件版本发布，关于CANN软件版本与本项目标签的对应关系请参阅[release仓库](https://gitcode.com/cann/release-management)中的相应版本说明 。
请注意，为确保您的源码定制开发顺利进行，请选择配套的CANN版本与Gitcode标签源码，使用master分支可能存在版本不匹配的风险。

## ⚡️快速入门

若您希望**从零到一了解并快速体验项目**，请访问如下文档。可以先了解项目算子信息，再尝试算子调用、开发、贡献等。

1. [算子列表](docs/zh/op_list.md)：项目全量算子信息，方便快速查询。
2. [QuickStart](QUICKSTART.md)：**基于WebIDE或Docker环境**提供极简快速上手指南，包括搭建环境、编译部署、算子调用/开发/调试、贡献等。

    > **说明**：无论是WebIDE或Docker环境，默认提供最新商发版本CANN软件包，目前是CANN 8.5.0。若您想手动安装CANN包或体验master分支最新能力等，请参考[学习教程](#学习教程)中的步骤，完成环境搭建、编译执行、算子开发等操作。

## 📖学习教程

若您已完成**快速入门**学习，对本项目有了一定认知，并希望**深入了解和体验项目**，请访问如下文档。

这些文档提供了多样化场景和更全面的操作指导，方便您应用于各种AI业务场景。

1. [环境部署](docs/zh/context/quick_install.md)：**基础环境搭建**指南，提供多种场景下CANN软件包安装方法，例如Docker部署、手动安装等。
2. [算子调用](docs/zh/invocation/quick_op_invocation.md)：算子**源码编译和执行**指南，提供多种编译算子包以及运行算子的方法（包括执行算子样例和UT）等。
3. [算子开发](docs/zh/develop/aicore_develop_guide.md)：基于项目工程**开发新算子**指南，从零学习创建算子工程、实现Tiling和Kernel核心交付件等。
4. [算子调试调优](docs/zh/debug/op_debug_prof.md)：提供常见的算子**调试和调优**方法，如DumpTensor、msProf、Simulator等。

除了上述指南，还提供了其他文档例如[算子调用方式](docs/zh/invocation/op_invocation.md)、术语概念、build参数介绍等，全量文档请访问[docs](docs/README.md)。

## 🔍目录结构

关键目录如下，详细目录介绍参见[项目目录](./docs/zh/context/dir_structure.md)。

```text
├── cmake                          # 项目工程编译目录
├── common                         # 项目公共头文件和公共源码
├── attention                      # attention类算子
│   ├── flash_attention_score      # flash_attention_score算子所有交付件
│   │   ├── CMakeLists.txt         # 算子编译配置文件
│   │   ├── docs                   # 算子说明文档
│   │   ├── examples               # 算子使用示例
│   │   ├── op_host                # 算子信息库、Tiling、InferShape相关实现目录
│   │   ├── op_kernel              # 算子Kernel目录
│   │   └── README.md              # 算子说明文档
│   ├── ...
│   └── CMakeLists.txt             # 算子编译配置文件
├── docs                           # 项目文档介绍
├── examples                       # 端到端算子开发和调用示例
├── experimental                   # 用户自定义算子存放目录
├── ...
├── moe                            # moe类算子
├── posembedding                   # posembedding类算子
├── scripts                        # 脚本目录，包含自定义算子、Kernel构建相关配置文件
├── tests                          # 测试工程目录
├── CMakeLists.txt
├── README.md
├── build.sh                       # 项目工程编译脚本
├── install_deps.sh                # 安装依赖包脚本
└── requirements.txt               # 本项目需要的第三方依赖包
```

## 💬相关信息

- [贡献指南](CONTRIBUTING.md)
- [安全声明](SECURITY.md)
- [许可证](LICENSE)
- [所属SIG](https://gitcode.com/cann/community/tree/master/CANN/sigs/ops-transformer)

## 🤝联系我们

本项目功能和文档正在持续更新和完善中，建议您关注最新版本。

- **问题反馈**：通过GitCode[【Issues】](https://gitcode.com/cann/ops-transformer/issues)提交问题。
- **社区互动**：通过GitCode[【讨论】](https://gitcode.com/cann/ops-transformer/discussions)参与交流。
- **技术专栏**：通过GitCode[【Wiki】](https://gitcode.com/cann/ops-transformer/wiki)获取技术文章，如系列化教程、优秀实践等。

    |技术专题|样例|
    |----|----|
    |算子性能优化实践|[FA算子性能优化实践和效果分析](https://gitcode.com/cann/ops-transformer/wiki/FA%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%E5%92%8C%E6%95%88%E6%9E%9C%E5%88%86%E6%9E%90.md)|
    