# GroupedMatmulSwigluQuantV2：源码快照与复现入口

本文重点：A8W8 + NZ 权重、TilingKey=3；Cube 128×256×128 基本块，整行就绪后的 Vector dequant/SwiGLU/RowMax/quant。

`upstream/` 里保存的是上游正式 `v9.0.0` 标签（提交 `afe72144f9f2ac8441929035795db88a111b30c5`）的原始文件，**不是伪代码或 API 调用包装**。本目录共保存 42 个上游文件，约 523.1 KiB。原始版权头和 [CANN Open Software License Agreement Version 2.0](upstream/LICENSE) 均保留。

文章主要研究 TilingKey=3 的 A8W8 fusion。为保留入口文件引用关系，快照也保留 A2/A3 的 A8W4/A4W4 源码；这些分支没有被文章逐项拆解。

保留 Kernel、Host Tiling / 算子注册 / aclnn 实现、A2/A3 调用示例，以及根 build.sh 和依赖配置等关键证据。主动排除 `arch35/` 的 950 实现、测试工程和其他无关算子。部分共享头文件仍保留上游编译条件下的 950 include；**快照不能作为独立工程直接编译**，编译需要固定提交的完整 ops-transformer 工程与 CANN。

- [原始文件 SHA256 清单](source-manifest.json)：记录每份文件的上游路径、大小、哈希及固定提交 URL。
- [案例正文](../docs/post.md)：定位主循环、布局、同步和参数选择。
- [复现步骤与未运行边界](../docs/reproduce.md)：完整工程构建、调用、数值与性能验证的区别。
- [证据与版本记录](../docs/sources.md)。

执行 `bash reproduce.sh verify` 可离线校验快照。`fetch` 仅获取两个固定源码仓库；`build` 必须先准备本地依赖并配置 CANN 9.0.0；`run-example` 在安装已构建的自定义包后，编译并调用本文指定的单个 A2/A3 样例。

本机是 macOS、没有 NPU。本目录完成了源码和脚本的静态核对，**没有声称 NPU 编译、数值验证或性能测量成功**。
