# FlashAttention反向SAMEAB/BN2基础Cube API与CV流水：原始源码快照

范围：A2/A3训练产品的arch32 SAMEAB/BN2；基础LoadData/Mmad/Fixpipe、L1输入复用、三任务CV流水与确定性归约。

官方 `ops-transformer v9.0.0`，提交 `afe72144f9f2ac8441929035795db88a111b30c5`。本目录保存 69 份上游原文件，约 2415.4 KiB，包括实际Kernel、Host Tiling/注册、指定API与调用样例，不用伪代码或包装器替代内核。

上游原始版权头和 [CANN Open Software License Agreement Version 2.0](upstream/LICENSE) 保留。逐文件路径、SHA256、大小和固定上游URL见 [source-manifest.json](source-manifest.json)。

刻意排除950的arch35、tests、配置知识库等。共享入口仍保留原始编译条件/其他实现引用，因此阅读快照**不是独立可编译工程**；完整复现必须获取固定仓库和实际CMake依赖。

- [正文](../docs/post.md)：具体数据切分、loader、资源、同步和取舍。
- [复现](../docs/reproduce.md)：固定源码、离线依赖、自定义包构建和样例。
- [来源](../docs/sources.md)、[元数据](../docs/metadata.json)。

本机macOS无NPU；只完成静态源码/脚本检查，没有NPU编译、数值测试或性能实测。`verify`离线校验，`fetch`只取固定源码，`build`要求CANN 9.0.0/Linux和预先准备的离线归档，`run-example`要求测试环境已装入自定义算子包。所有build源与opbase检查commit和dirty状态，防止同HEAD修改被误记成原始版本。

支持A2/A3训练系列，产品表不支持相应推理系列。官方普通样例用常量Softmax统计，只能作为API调用样例；梯度验证必须使用真实前向结果/Softmax max/sum及同一Dropout mask。SAMEAB一般路径不支持FP32，默认FP32示例不覆盖目标模板。
