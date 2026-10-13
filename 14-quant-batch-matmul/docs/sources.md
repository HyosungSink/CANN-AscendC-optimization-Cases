# 来源与证据记录

核对日期：2026-10-08（Asia/Shanghai）。只使用官方源码和官方CANN发布说明。

| 项目 | 固定证据 |
| --- | --- |
| 上游 | [cann/ops-nn](https://gitcode.com/cann/ops-nn) |
| 正式标签与提交 | `v9.0.0` → `0413e303e6cd1986b592c3e039f2579ac0f10b62`；git ls-remote与实际clone后HEAD一致 |
| CANN配套 | [9.0.0发布说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) 将ops-nn 9.0.0对应v9.0.0且配套CANN 9.0.0 |
| 硬件 | [README](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/quant_batch_matmul_v3/README.md) 产品表含A2/A3；[def注册](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/quant_batch_matmul_v3/op_host/quant_batch_matmul_v3_def.cpp) 第76–77行有ascend910b/ascend910_93 |
| opbase实际依赖 | [opbase.cmake](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/cmake/third_party/opbase.cmake) 固定 `ec88904b611b04f9b23b529aee66c78bb7cf6140`，不替换为master或别的标签 |
| 构建与样例 | [官方构建说明](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/docs/zh/invocation/quick_op_invocation.md)；[指定样例](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/quant_batch_matmul_v3/examples/test_aclnn_quant_matmul_v3_at2_at3.cpp) |
| 许可 | [原始LICENSE](../code/upstream/LICENSE)，版权头保留 |

## 核心源码

| 固定上游链接 | 本地保存文件 |
| --- | --- |
| [op_kernel/quant_batch_matmul_v3_pertoken_opt.h](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_opt.h) | [本地原文件](../code/upstream/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3_pertoken_opt.h) |
| [op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp) | [本地原文件](../code/upstream/matmul/quant_batch_matmul_v3/op_host/op_tiling/quant_batch_matmul_v3_tiling.cpp) |
| [op_kernel/quant_batch_matmul_v3.cpp](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.cpp) | [本地原文件](../code/upstream/matmul/quant_batch_matmul_v3/op_kernel/quant_batch_matmul_v3.cpp) |

## 可据此作出的结论

BmmDequantPertokenOpt：小M、50MiB workspace限制、scale外积预计算、15次分组同步。

版本和硬件的依据为发布配套与Host注册；文章的阈值、缓存与流水事实来自上述原始代码。性能收益方向属于对机制的分析，**没有声称量化加速倍数**。

| 验证层次 | 当前状态 |
| --- | --- |
| 固定版本/硬件源码核对 | 已完成 |
| SHA256与脚本静态校验 | 已完成 |
| NPU编译与运行 | 未执行，本机macOS无NPU |
| NPU随机数值回归 | 未执行 |
| NPU单算子benchmark | 未执行 |
| 官方公开优化前后耗时 | 在检阅材料中未找到固定版本可对应的AB数据 |
| 模型TPOT/tokens/s | 未用作本篇性能证据 |

示例文件名明确面向A2/A3；模板选择依赖pertoken、输出dtype、shape、workspace等条件。调用成功不等于已运行opt模板，必须检查tiling/profiler。

“CANN 9.0.0及以上”在此指选用9.0.0正式配套案例；升级后需使用相应发布源码重新验证，本文不提供对任意未来版本的ABI保证。
