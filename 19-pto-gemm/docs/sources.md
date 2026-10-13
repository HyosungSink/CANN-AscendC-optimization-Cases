# 固定来源和版本证据

收集日期：2026-10-08（Asia/Shanghai）。上游代码许可：CANN Open Software License Agreement Version 2.0；原始版权及 LICENSE 保留。文章为本合集原创源码解析。

| 用途 | 固定来源 | 本地保存 |
|---|---|---|
| 正式v9.0.0与CANN9.0.0配套 | [release-management](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) | [发布说明](evidence/release-notes-9.0.0.md) |
| 原始算子/案例和硬件范围 | [上游](https://gitcode.com/cann/pto-isa/blob/377526952e95c39f9149cb94d261eba9e394f0d4/kernels/manual/a2a3/gemm_performance/README_zh.md) | [原文](../code/upstream/kernels/manual/a2a3/gemm_performance/README_zh.md) |
| 包版本9.0.0 | [version.cmake](https://gitcode.com/cann/pto-isa/blob/377526952e95c39f9149cb94d261eba9e394f0d4/version.cmake) | [文件](../code/upstream/version.cmake) |
| 核心实现/Host | [上游](https://gitcode.com/cann/pto-isa/blob/377526952e95c39f9149cb94d261eba9e394f0d4/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp) | [文件](../code/upstream/kernels/manual/a2a3/gemm_performance/gemm_performance_kernel.cpp) |
| 核心实现/Host | [上游](https://gitcode.com/cann/pto-isa/blob/377526952e95c39f9149cb94d261eba9e394f0d4/kernels/manual/a2a3/gemm_performance/main.cpp) | [文件](../code/upstream/kernels/manual/a2a3/gemm_performance/main.cpp) |

源码固定tag存在与支持范围是版本证据；公开性能表的精确测量CANN和历史commit未注明，不能从tag倒推。没有本机NPU测量。

全部文件来源和 SHA-256 见 [`source-manifest.json`](../code/source-manifest.json)。
