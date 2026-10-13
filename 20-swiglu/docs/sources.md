# 来源与版本证据

核对日期为 2026年10月8日。ops-nn 正式 `v9.0.0` 标签解析为 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。[官方发布映射](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)明确配套 CANN 9.0.0。

- [产品及公式](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/activation/swi_glu/README.md)：A2/A3，指定轴二分的 SwiGLU。
- [Host Tiling](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/activation/swi_glu/op_host/swi_glu_tiling.cpp)：512B 打包、UB 字节预算、单/双缓冲选择与 stride 限制。
- [低精度 Kernel](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/activation/swi_glu/op_kernel/swi_glu_bf16.hpp)：FP32 计算、临时空间复用及 Cast。
- [算子注册](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/activation/swi_glu/op_host/swi_glu_def.cpp)：ascend910b/ascend910_93 配置。

固定版本与核心源码链接见本页来源表或 [来源说明](sources.md)，快照保留原始版权和许可证；逐文件 SHA256 见 [整包清单](../../bundle-manifest.json)。
