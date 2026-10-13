# 来源与证据记录

核对日期：2026-10-08（Asia/Shanghai）。只使用官方源码和官方CANN发布说明。

| 项目 | 固定证据 |
| --- | --- |
| 上游 | [cann/ops-nn](https://gitcode.com/cann/ops-nn) |
| 正式标签与提交 | `v9.0.0` → `0413e303e6cd1986b592c3e039f2579ac0f10b62`；git ls-remote与实际clone后HEAD一致 |
| CANN配套 | [9.0.0发布说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) 将ops-nn 9.0.0对应v9.0.0且配套CANN 9.0.0 |
| 硬件 | [README](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/weight_quant_batch_matmul_v2/README.md) 产品表含A2/A3；[def注册](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/weight_quant_batch_matmul_v2/op_host/weight_quant_batch_matmul_v2_def.cpp) 第212–213行有ascend910b/ascend910_93 |
| opbase实际依赖 | [opbase.cmake](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/cmake/third_party/opbase.cmake) 固定 `ec88904b611b04f9b23b529aee66c78bb7cf6140`，不替换为master或别的标签 |
| 构建与样例 | [官方构建说明](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/docs/zh/invocation/quick_op_invocation.md)；[指定样例](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/weight_quant_batch_matmul_v2/examples/test_aclnn_weight_quant_batch_matmul_v2.cpp) |
| 许可 | [原始LICENSE](../code/upstream/LICENSE)，版权头保留 |

## 核心源码

| 固定上游链接 | 本地保存文件 |
| --- | --- |
| [op_kernel/weight_quant_batch_matmul_v2_custom_nz_splitk.h](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom_nz_splitk.h) | [本地原文件](../code/upstream/matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2_custom_nz_splitk.h) |
| [op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.cpp](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.cpp) | [本地原文件](../code/upstream/matmul/weight_quant_batch_matmul_v2/op_host/op_tiling/weight_quant_batch_matmul_v2_tiling_custom_nz_splitk.cpp) |
| [op_kernel/weight_quant_batch_matmul_v2.cpp](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2.cpp) | [本地原文件](../code/upstream/matmul/weight_quant_batch_matmul_v2/op_kernel/weight_quant_batch_matmul_v2.cpp) |

## 可据此作出的结论

NZ低比特权重，Vector反量化与Cube四槽消费、K分核、A全载和原子后处理。

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

默认样例是ND权重，不会自动覆盖本文NZ Split-K。需要按aclnnWeightQuantBatchMatmulNz契约准备真实NZ布局，M在65–256、A不转置/B转置、per-channel scale，且非确定性；它是算子内部专用模板，不是通用W8W4保证。

“CANN 9.0.0及以上”在此指选用9.0.0正式配套案例；升级后需使用相应发布源码重新验证，本文不提供对任意未来版本的ABI保证。
