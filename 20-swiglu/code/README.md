# swi_glu 源码快照

本目录保存 `ops-nn` 正式 `v9.0.0` 标签的 `activation/swi_glu` 源码，完整 commit 为 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。已排除 950 专用 `arch35` 实现和其他硬件的二进制配置；公共定义中的其他硬件注册仍按上游原样保留。核心源码、Host Tiling、aclnn 示例和 UT 均保留原始版权。逐文件 URL、大小和 SHA256 见 [source-manifest.json](source-manifest.json)。

快照用于离线阅读，编译需完整上游工程、其固定依赖、Linux 与配套 CANN Toolkit。`reproduce.sh fetch` 获取完整固定工程，`reproduce.sh build` 生成算子包，不执行安装。详细命令见 [复现说明](../docs/reproduce.md)。

许可为 [CANN Open Software License Agreement Version 2.0](upstream/LICENSE)。
