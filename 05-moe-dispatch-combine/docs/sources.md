# 来源与证据边界

所有源文件来自上游 MIT 项目；原文件版权与 LICENSE 已保留。文章分析为本合集原创整理。

固定仓库：`https://github.com/sgl-project/sgl-kernel-npu`

固定 commit：`653e519cf9556b0cc8d9e7966aa4551eb65143ae`

收集时间：2026-10-08（Asia/Shanghai）。

| 用途 | 原始固定来源 | 本地快照 |
|---|---|---|
| 版本、硬件与API | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/python/deep_ep/README.md) | [文件](../code/upstream/python/deep_ep/README.md) |
| A2/A3正式9.0.0/9.1.0测试矩阵 | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/.github/workflows/pr-test-deepep-npu.yml) | [文件](../code/upstream/.github/workflows/pr-test-deepep-npu.yml) |
| 公开算子级性能表（测量CANN未知） | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/README.md) | [文件](../code/upstream/README.md) |
| A2配套与通信拓扑 | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/python/deep_ep/doc/A2_DEEPEP.md) | [文件](../code/upstream/python/deep_ep/doc/A2_DEEPEP.md) |
| 低延迟接口、窗口容量、INT8解释 | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/python/deep_ep/doc/LOW_LATENCY_API.md) | [文件](../code/upstream/python/deep_ep/doc/LOW_LATENCY_API.md) |
| A3 Dispatch核心 | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/csrc/deepep/ops/op_kernel/moe_distribute_dispatch_v2.h) | [文件](../code/upstream/csrc/deepep/ops/op_kernel/moe_distribute_dispatch_v2.h) |
| A3 Combine核心 | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/csrc/deepep/ops/op_kernel/moe_distribute_combine_v2.h) | [文件](../code/upstream/csrc/deepep/ops/op_kernel/moe_distribute_combine_v2.h) |
| Host tiling | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/csrc/deepep/ops/op_host/moe_distribute_dispatch_v2_tiling.cpp) | [文件](../code/upstream/csrc/deepep/ops/op_host/moe_distribute_dispatch_v2_tiling.cpp) |
| C++桥接与量化模式 | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/csrc/deepep/deep_ep.cpp) | [文件](../code/upstream/csrc/deepep/deep_ep.cpp) |
| 正确性和单独kernel profiler测试 | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/tests/python/deepep/test_low_latency.py) | [文件](../code/upstream/tests/python/deepep/test_low_latency.py) |
| Event和Profiler计时实现 | [上游](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/tests/python/deepep/utils.py) | [文件](../code/upstream/tests/python/deepep/utils.py) |

性能表是 A3 的绝对延迟，不是与基线比较的加速倍数；测量版本未单列。当前版本兼容性和CI不能倒推出历史测量版本。A2支持证据也不能迁移成A2有相同延迟。

固定版本与核心源码链接见本页来源表或 [来源说明](sources.md)，快照保留原始版权和许可证；逐文件 SHA256 见 [整包清单](../../bundle-manifest.json)。
