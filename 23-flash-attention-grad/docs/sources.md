# 来源与证据层级

核对日期：2026-10-08（Asia/Shanghai）。正式`v9.0.0`，非beta/rc。

| 证据 | 固定来源 |
| --- | --- |
| 上游 | [cann/ops-transformer](https://gitcode.com/cann/ops-transformer)，commit `afe72144f9f2ac8441929035795db88a111b30c5` |
| CANN版本配套 | [release-management 9.0.0/release-notes.md](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)，明确ops-transformer v9.0.0配套CANN 9.0.0 |
| 产品 | [README](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/flash_attention_score_grad/README.md)，支持范围为Atlas A2 training series, Atlas A3 training series |
| 实际Host硬件注册 | [def文件](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/flash_attention_score_grad/op_host/flash_attention_score_grad_def.cpp) 第501–502行：ascend910b、ascend910_93 |
| 实际opbase依赖 | [opbase.cmake](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/cmake/third_party/opbase.cmake)，commit `929902e577bb809a377a6ab17cb7b00a4dfddc1a` |
| 构建指引 | [quick_op_invocation.md](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/docs/zh/invocation/quick_op_invocation.md) |
| 调用样例 | [固定源码示例](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v2.cpp) |
| 许可 | [上游LICENSE](../code/upstream/LICENSE)；源码版权头保留 |

## 核心实现

| 固定上游链接 | 保存文件 |
| --- | --- |
| [op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h) | [本地原文件](../code/upstream/attention/flash_attention_score_grad/op_kernel/arch32/flash_attention_score_grad_s1s2_bn2gs1s2_sab.h) |
| [op_kernel/arch32/basic_modules/cube_op_bn2.h](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/flash_attention_score_grad/op_kernel/arch32/basic_modules/cube_op_bn2.h) | [本地原文件](../code/upstream/attention/flash_attention_score_grad/op_kernel/arch32/basic_modules/cube_op_bn2.h) |
| [op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp) | [本地原文件](../code/upstream/attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2gs1s2_sab.cpp) |
| [op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2.cpp](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2.cpp) | [本地原文件](../code/upstream/attention/flash_attention_score_grad/op_host/arch32/flash_attention_score_grad_tiling_s1s2_bn2.cpp) |
| [op_kernel/flash_attention_score_grad.cpp](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp) | [本地原文件](../code/upstream/attention/flash_attention_score_grad/op_kernel/flash_attention_score_grad.cpp) |

## 性能证据

发布说明第149行报告SAMEAB/BN2基础API重写相对CANN 8.5.0，泛化测试平均优化1%+ / 10%+。原条目未列逐shape绝对耗时、完整测试集或硬件清单，因此属于**官方版本级模板汇总**，不等价于任意A2/A3设备上每个shape的AB数据，也不是L1复用的单开关消融。本文保留原始百分比表述，不推成未经定义的加速倍数。

| 验证层次 | 当前状态 |
| --- | --- |
| 正式tag与A2/A3实际Kernel核对 | 已完成 |
| SHA256、JSON、链接、shell语法 | 已完成 |
| NPU编译/运行/随机数值比对 | 未执行，本机macOS无NPU |
| NPU单shape benchmark | 未执行 |
| 端到端TPOT/tokens/s | 未用作算子性能证据 |

支持A2/A3训练系列，产品表不支持相应推理系列。官方普通样例用常量Softmax统计，只能作为API调用样例；梯度验证必须使用真实前向结果/Softmax max/sum及同一Dropout mask。SAMEAB一般路径不支持FP32，默认FP32示例不覆盖目标模板。

更高CANN版本应取匹配发布源码重新验证；本目录是正式9.0.0案例，不对任意未来ABI作未经测试的保证。
