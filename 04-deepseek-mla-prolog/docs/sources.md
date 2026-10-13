# 来源、适用性与证据级别

核对日期：**2026-10-08（Asia/Shanghai）**。所有技术事实取自官方源码与发布说明，文章中的优化收益解释是基于实现依赖的分析。

| 记录 | 固定值 / 证据 |
| --- | --- |
| 官方上游仓库 | [cann/ops-transformer](https://gitcode.com/cann/ops-transformer) |
| 正式发布标签 | `v9.0.0`；不是 beta / rc |
| 源码提交 | `afe72144f9f2ac8441929035795db88a111b30c5`（用 `git ls-remote` 和实际 fetch 后 `git rev-parse HEAD` 核对） |
| 版本映射 | [CANN 9.0.0 release-notes.md](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) 的独立子包配套表明确列出 ops-transformer `v9.0.0` 配套 CANN 9.0.0 |
| 目标硬件 | [算子 README](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/mla_prolog_v3/README.md) 列出 Atlas A2/A3；Host def 第 171–172 行注册 `ascend910b`、`ascend910_93` |
| SoC 名称对应 | [官方构建说明](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/docs/zh/invocation/quick_op_invocation.md)：A2=`ascend910b`，A3=`ascend910_93` |
| 实际构建依赖 | [cmake/third_party/opbase.cmake](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/cmake/third_party/opbase.cmake) 固定 opbase `929902e577bb809a377a6ab17cb7b00a4dfddc1a` |
| 许可 | [上游 LICENSE](../code/upstream/LICENSE)，所有快照保留原始版权头 |

注意：release-management 同时列出 opbase `v9.0.0` 标签，但本算子工程 CMake 固定了不同的 opbase 源码提交。复现以工程实际依赖 `929902e577bb809a377a6ab17cb7b00a4dfddc1a` 为准，不擅自替换为当前 master 或另一个 tag。

## 必读源码

| 上游固定提交链接 | 本地原文件 |
| --- | --- |
| [op_kernel/mla_prolog_v3.cpp](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp) | [本地快照](../code/upstream/attention/mla_prolog_v3/op_kernel/mla_prolog_v3.cpp) |
| [op_host/CMakeLists.txt](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/mla_prolog_v3/op_host/CMakeLists.txt) | [本地快照](../code/upstream/attention/mla_prolog_v3/op_host/CMakeLists.txt) |
| [op_host/mla_prolog_v3_tiling_register.cpp](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/mla_prolog_v3/op_host/mla_prolog_v3_tiling_register.cpp) | [本地快照](../code/upstream/attention/mla_prolog_v3/op_host/mla_prolog_v3_tiling_register.cpp) |
| [op_host/mla_prolog_v3_def.cpp](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp) | [本地快照](../code/upstream/attention/mla_prolog_v3/op_host/mla_prolog_v3_def.cpp) |
| [A2/A3 调用样例](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/mla_prolog_v3/examples/test_aclnn_mla_prolog_v3.cpp) | [本地样例](../code/upstream/attention/mla_prolog_v3/examples/test_aclnn_mla_prolog_v3.cpp) |

MLA 额外核心依赖：

- `attention/mla_prolog/op_kernel/kernel_mla_prolog_split_n.h`：`Process/AicProcess/AivProcess`、A 全载、Split-N CV 流水与 Cache。
- `attention/mla_prolog/op_kernel/service_matmul.h`：L1/L0 缓冲、ND/NZ 搬运、Matmul。
- `attention/mla_prolog/op_kernel/service_rms_norm.h`、`arch32/rms_norm.h`：A2/A3 RMSNorm 与动态量化。
- `attention/mla_prolog/op_host/mla_prolog_tiling.cpp`：模板与 shape 条件。

## 性能与验证证据

| 类型 | 当前证据 | 可以支持的结论 |
| --- | --- | --- |
| 硬件与 CANN 配套 | 正式 tag + 官方配套表 + Host 硬件注册 | 满足本合集 CANN 9.0.0、A2/A3 源码范围 |
| 具体优化机制 | Kernel 与 Host Tiling 原始源码 | 能核对布局、模板条件、同步、缓存和融合逻辑 |
| 单算子优化前后公开耗时 | **未在这些官方资料中找到** | 不给出加速倍数，不借用模型 TPOT 或 throughput |
| 本机 NPU 编译 / 运行 | **未执行**；本机是 macOS，没有 NPU | 不能声称运行通过或达到性能目标 |
| 本机静态检查 | SHA256 快照校验、脚本语法和关键路径存在性 | 确认保存完整且与固定源码一致，不等于内核正确性验证 |

官方样例使用 B=8、S=1、He=7168、Hcq=1536、head=128、D=128、Hckv=512、Dr=64，主要初始化为零。它不会自动覆盖半量化 token=1 算力分组分支，也不构成数值回归或性能验证。

后续用户在真实 NPU 上测量时，至少记录具体设备/SoC、CANN Toolkit 和 ops 子包版本、HDK、完整提交、输入 shape/dtype/layout、有效 token/专家分布、数值误差和 device 时间。对于更新的 CANN，应使用对应发布标签重新核对，而不是把“9.0.0 及以上”当成未经测试的无限兼容保证。
