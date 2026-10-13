# 固定来源和版本证据

收集日期：2026-10-08（Asia/Shanghai）。上游代码许可：CANN Open Software License Agreement Version 2.0；原始版权及 LICENSE 保留。文章为本合集原创源码解析。

| 用途 | 固定来源 | 本地保存 |
|---|---|---|
| 正式v9.0.0与CANN9.0.0配套 | [release-management](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md) | [发布说明](evidence/release-notes-9.0.0.md) |
| 原始算子/案例和硬件范围 | [上游](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/posembedding/rotary_position_embedding/README.md) | [原文](../code/upstream/posembedding/rotary_position_embedding/README.md) |
| 核心实现/Host | [上游](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/posembedding/rotary_position_embedding/op_host/rope_rotate_half_tiling.cpp) | [文件](../code/upstream/posembedding/rotary_position_embedding/op_host/rope_rotate_half_tiling.cpp) |
| 核心实现/Host | [上游](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/posembedding/rotary_position_embedding/op_kernel/rotate_interleaved_split_bs.h) | [文件](../code/upstream/posembedding/rotary_position_embedding/op_kernel/rotate_interleaved_split_bs.h) |
| 核心实现/Host | [上游](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/posembedding/rotary_position_embedding/op_kernel/rotate_half_bf16.h) | [文件](../code/upstream/posembedding/rotary_position_embedding/op_kernel/rotate_half_bf16.h) |

源码固定tag存在与支持范围是版本证据；公开性能表的精确测量CANN和历史commit未注明，不能从tag倒推。没有本机NPU测量。

固定版本与核心源码链接见本页来源表或 [来源说明](sources.md)，快照保留原始版权和许可证；逐文件 SHA256 见 [整包清单](../../bundle-manifest.json)。
