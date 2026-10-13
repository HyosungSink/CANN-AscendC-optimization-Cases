# RmsNorm 的来源与证据

固定版本与核心源码链接见本页来源表或 [来源说明](sources.md)，快照保留原始版权和许可证；逐文件 SHA256 见 [整包清单](../../bundle-manifest.json)。

| 一手来源 | 支持的结论 |
|---|---|
| [CANN 9.0.0 release notes](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) | 独立子包配套表列 ops-nn 9.0.0、源码标签 v9.0.0、配套 CANN 9.0.0；另存 evidence/cann-9.0.0-release-notes.md |
| [算子 README](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/rms_norm/README.md) | A2/A3 支持、参数语义、dtype 和示例 |
| [A2 配置](https://gitcode.com/cann/ops-nn/tree/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/rms_norm/op_host/config/ascend910b) 与 [A3 配置](https://gitcode.com/cann/ops-nn/tree/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/rms_norm/op_host/config/ascend910_93) | 对应目标有非 950 内核配置，已随快照保存 |
| [op_host/rms_norm_tiling.cpp](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/rms_norm/op_host/rms_norm_tiling.cpp) | 正文中具体阈值、缓冲与函数分析的依据 |
| [op_kernel/rms_norm_merge_n.h](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/rms_norm/op_kernel/rms_norm_merge_n.h) | 正文中具体阈值、缓冲与函数分析的依据 |
| [op_kernel/rms_norm_split_d.h](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/rms_norm/op_kernel/rms_norm_split_d.h) | 正文中具体阈值、缓冲与函数分析的依据 |
| [op_kernel/rms_norm_single_row.h](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/norm/rms_norm/op_kernel/rms_norm_single_row.h) | 正文中具体阈值、缓冲与函数分析的依据 |
| [算子调用指南](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/docs/zh/invocation/quick_op_invocation.md) | 单算子自定义包、cust example、Host/API UT 路线 |
| [opbase CMake](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/cmake/third_party/opbase.cmake) | 固定依赖提交 ec88904b611b04f9b23b529aee66c78bb7cf6140 |
| [LICENSE](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/LICENSE) | CANN Open Software License Agreement Version 2.0，版权和许可原样保留 |

`git ls-remote https://gitcode.com/cann/ops-nn.git refs/tags/v9.0.0` 返回 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。正式 tag 优于同名移动分支，快照使用完整 commit 链接。

本文属于真实实现的源码实战解析，性能证据级别是**机制已核对，A/B 未复测**。没有将架构支持当作任意 shape 的性能证明，没有引用 950 独占路径的数值。`metadata.json` 的 public_ab_benchmark 和 measured_locally 均为 false。
