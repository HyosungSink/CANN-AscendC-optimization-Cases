# 来源与证据层级

核对日期：2026-10-08（Asia/Shanghai）。正式`v9.0.0`，非beta/rc。

| 证据 | 固定来源 |
| --- | --- |
| 上游 | [cann/ops-nn](https://gitcode.com/cann/ops-nn)，commit `0413e303e6cd1986b592c3e039f2579ac0f10b62` |
| CANN版本配套 | [release-management 9.0.0/release-notes.md](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)，明确ops-nn v9.0.0配套CANN 9.0.0 |
| 产品 | [README](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/deformable_conv2d/README.md)，支持范围为Atlas A2 training/inference series, Atlas A3 training/inference series |
| 实际Host硬件注册 | [def文件](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/deformable_conv2d/op_host/deformable_conv2d_def.cpp) 第60–61行：ascend910b、ascend910_93 |
| 实际opbase依赖 | [opbase.cmake](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/cmake/third_party/opbase.cmake)，commit `ec88904b611b04f9b23b529aee66c78bb7cf6140` |
| 构建指引 | [quick_op_invocation.md](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/docs/zh/invocation/quick_op_invocation.md) |
| 调用样例 | [固定源码示例](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/deformable_conv2d/examples/test_aclnn_deformable_conv2d.cpp) |
| 许可 | [上游LICENSE](../code/upstream/LICENSE)；源码版权头保留 |

## 核心实现

| 固定上游链接 | 保存文件 |
| --- | --- |
| [op_kernel/deformable_conv2d_base.h](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/deformable_conv2d/op_kernel/deformable_conv2d_base.h) | [本地原文件](../code/upstream/conv/deformable_conv2d/op_kernel/deformable_conv2d_base.h) |
| [op_kernel/deformable_conv2d_base.cpp](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/deformable_conv2d/op_kernel/deformable_conv2d_base.cpp) | [本地原文件](../code/upstream/conv/deformable_conv2d/op_kernel/deformable_conv2d_base.cpp) |
| [op_host/deformable_conv2d_tiling.cpp](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/deformable_conv2d/op_host/deformable_conv2d_tiling.cpp) | [本地原文件](../code/upstream/conv/deformable_conv2d/op_host/deformable_conv2d_tiling.cpp) |
| [op_host/op_api/aclnn_deformable_conv2d.cpp](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/conv/deformable_conv2d/op_host/op_api/aclnn_deformable_conv2d.cpp) | [本地原文件](../code/upstream/conv/deformable_conv2d/op_host/op_api/aclnn_deformable_conv2d.cpp) |

## 性能证据

在检阅的官方资料中未找到固定版本对应的公开AB单算子耗时；不提供虚构加速比。

| 验证层次 | 当前状态 |
| --- | --- |
| 正式tag与A2/A3实际Kernel核对 | 已完成 |
| SHA256、JSON、链接、shell语法 | 已完成 |
| NPU编译/运行/随机数值比对 | 未执行，本机macOS无NPU |
| NPU单shape benchmark | 未执行 |
| 端到端TPOT/tokens/s | 未用作算子性能证据 |

Conv2DV2/ExtendConv2D正式tag只支持950，未选用。当前aclnnDeformableConv2d要求modulated=true；A2/A3外部FP16/BF16会先Cast到FP32，bias和输出变换位于API后处理。默认样例Cin=6、5×5核，不覆盖对齐小C专用路径。

更高CANN版本应取匹配发布源码重新验证；本目录是正式9.0.0案例，不对任意未来ABI作未经测试的保证。
