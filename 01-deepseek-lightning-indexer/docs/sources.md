# LightningIndexer 的来源与证据

固定版本与核心源码链接见本页来源表或 [来源说明](sources.md)，快照保留原始版权和许可证；逐文件 SHA256 见 [整包清单](../../bundle-manifest.json)。

| 来源 | 支持的结论 |
|---|---|
| [CANN 9.0.0 正式 release notes](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) | 独立子包配套表将 ops-transformer 9.0.0 映射为 v9.0.0，并列 CANN 9.0.0 配套；此文件另保存为 evidence/cann-9.0.0-release-notes.md |
| [算子 README](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/lightning_indexer/README.md) | 产品支持、dtype、layout、shape、接口语义与样例 |
| [Host Tiling](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/lightning_indexer/op_host/lightning_indexer_tiling.cpp) | `LIInfoParser::GetNpuInfo` 显式接受 ASCEND910B 和 ASCEND910_93。 以及模式/尺寸选择 |
| [lightning_indexer_kernel.h](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/lightning_indexer/op_kernel/lightning_indexer_kernel.h) | 本文逐函数分析的主来源 |
| [lightning_indexer_service_cube.h](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/lightning_indexer/op_kernel/lightning_indexer_service_cube.h) | 本文逐函数分析的主来源 |
| [lightning_indexer_service_vector.h](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/lightning_indexer/op_kernel/lightning_indexer_service_vector.h) | 本文逐函数分析的主来源 |
| [lightning_indexer_vector.h](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/lightning_indexer/op_kernel/lightning_indexer_vector.h) | 本文逐函数分析的主来源 |
| [调用指南](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/docs/zh/invocation/quick_op_invocation.md) | 指定单算子自定义包构建、cust example 与 UT 命令 |
| [opbase 依赖配置](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/cmake/third_party/opbase.cmake) | 固定 opbase 提交 929902e577bb809a377a6ab17cb7b00a4dfddc1a |
| [源码许可证](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/LICENSE) | CANN Open Software License Agreement Version 2.0；副本在 code/upstream/LICENSE |

## 版本固定核验

```bash
git ls-remote https://gitcode.com/cann/ops-transformer.git refs/tags/v9.0.0
# afe72144f9f2ac8441929035795db88a111b30c5 refs/tags/v9.0.0
```

9.0.0 分支核对时为 `a138580d3a6d43a334c5ddacb0032195f1b6a22d`。本材料选择正式 release notes 指向的 v9.0.0，避免移动分支；本算子的 Kernel/Host 在这两个提交间无差异，分支新增的英文 API 文档未用于正式标签快照。

## 性能证据的范围

本文解释具体优化机制、数据布局与可实施的消融，属于源码实战解析。未核实到该算子同时固定 CANN 9.0.0、非 950 设备、完整 shape、计时口径和原版基线的公开 A/B 表。`metadata.json` 明确记录 `public_ab_benchmark=false` 与 `measured_locally=false`。源码中启用了优化结构，不等于任意 shape 都已测得更快。

自行加入的 `benchmark.py` 只提供复现实验入口，没有保存示例性能值。正文的 MiB 数值由源码尺寸常量计算，并非峰值实测带宽或端到端收益。
