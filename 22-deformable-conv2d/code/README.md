# 可变形卷积的坐标复用、相邻访存合并与Cube融合：原始源码快照

范围：A2/A3的FP32主路径；2048采样点预算、小C二维DMA、标准坐标Adds复用、插值输入重排和GEMM。

官方 `ops-nn v9.0.0`，提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。本目录保存 19 份上游原文件，约 201.0 KiB，包括实际Kernel、Host Tiling/注册、指定API与调用样例，不用伪代码或包装器替代内核。

上游原始版权头和 [CANN Open Software License Agreement Version 2.0](upstream/LICENSE) 保留。逐文件路径、SHA256、大小和固定上游URL见 [source-manifest.json](source-manifest.json)。

刻意排除950的arch35、tests、配置知识库等。共享入口仍保留原始编译条件/其他实现引用，因此阅读快照**不是独立可编译工程**；完整复现必须获取固定仓库和实际CMake依赖。

- [正文](../docs/post.md)：具体数据切分、loader、资源、同步和取舍。
- [复现](../docs/reproduce.md)：固定源码、离线依赖、自定义包构建和样例。
- [来源](../docs/sources.md)、[元数据](../docs/metadata.json)。

本机macOS无NPU；只完成静态源码/脚本检查，没有NPU编译、数值测试或性能实测。`verify`离线校验，`fetch`只取固定源码，`build`要求CANN 9.0.0/Linux和预先准备的离线归档，`run-example`要求测试环境已装入自定义算子包。所有build源与opbase检查commit和dirty状态，防止同HEAD修改被误记成原始版本。

Conv2DV2/ExtendConv2D正式tag只支持950，未选用。当前aclnnDeformableConv2d要求modulated=true；A2/A3外部FP16/BF16会先Cast到FP32，bias和输出变换位于API后处理。默认样例Cin=6、5×5核，不覆盖对齐小C专用路径。
