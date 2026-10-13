# LightningIndexer 源码与复现工具

`upstream/attention/lightning_indexer/` 是官方 `cann/ops-transformer` 的原始完整算子目录，包含 Kernel、Host Tiling、InferShape、aclnn 包装、示例、UT 和参数说明。固定 `v9.0.0`，提交 `afe72144f9f2ac8441929035795db88a111b30c5`。源码没有经过修改，原版权头和仓库根 LICENSE 均保留。

另外保存上游构建入口、opbase 依赖配置和构建/调优文档作为查阅依据。这是一个算子源码快照，**不是可独立构建的完整 ops-transformer 工程**；构建需要 `reproduce.sh` 获取完整固定版本工程。SDK 的 Ascend C、matmul、Softmax、runtime/opdev 等头文件和库由配套 CANN 提供，`opbase` 由完整上游工程的 CMake 获取固定提交 `929902e577bb809a377a6ab17cb7b00a4dfddc1a`。不会从本快照目录直接编译。

支持 Atlas A2 与 Atlas A3 的训练/推理系列。A2 使用 ascend910b；A3 使用 ascend910_93。两者在 Host 的 GetNpuInfo 中都有显式接受条件。

- `source-manifest.json`：每个上游快照文件的 SHA256、字节数、原路径及固定提交链接。
- `reproduce.sh`：获取完整固定源码，按指定动作构建自定义算子包或运行上游 example/UT。默认只 fetch，不安装 CANN、不用 sudo、不自动安装生成包。
- `benchmark.py`：本次新增的 PyTorch 单算子检查与计时工具，BSND、完整有效长度、无 causal mask；参考计算在 CPU 上完成并且不计入 Kernel 时间。它不是上游实现，也不是本机实测记录。

先读 [复现说明](../docs/reproduce.md)。许可证文本在 [upstream/LICENSE](upstream/LICENSE)，适用范围保留上游条款。
