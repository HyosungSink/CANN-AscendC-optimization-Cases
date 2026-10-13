# 来源和适配边界

固定 tag：`v3.2.1`；commit：`2badfc89e70a9b7a5e88463a116c2feddce4b101`。收集日期：2026-10-08（Asia/Shanghai）。

上游代码与文档使用 MIT 许可证，原始版权和 `code/upstream/LICENSE` 保留。原始文件未修改；`code/examples/` 是透明标注的独立适配和 CPU golden。

| 证据 | 原始来源 | 本地保存 |
|---|---|---|
| 正式配套CANN9.0.0和A2/A3 | [来源](https://github.com/triton-lang/triton-ascend/releases/tag/v3.2.1) | [本地](upstream-release.json) |
| 真实单次load→循环tiling改写 | [来源](https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/docs/zh/migration_guide/performance_guidelines.md) | [本地](../code/upstream/docs/zh/migration_guide/performance_guidelines.md) |
| care_padding精确语义 | [来源](https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/python/triton/language/core.py) | [本地](../code/upstream/python/triton/language/core.py) |
| 自动multi-buffer编译选项 | [来源](https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/third_party/ascend/backend/compiler.py) | [本地](../code/upstream/third_party/ascend/backend/compiler.py) |
| 硬件支持说明 | [来源](https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/README.md) | [本地](../code/upstream/README.md) |
| multi-buffer接口约束 | [来源](https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/docs/zh/triton_api/Extension_Ops/multibuffer.md) | [本地](../code/upstream/docs/zh/triton_api/Extension_Ops/multibuffer.md) |

版本以正式 release notes 为依据；固定 quick_start 的8.5.0 Docker示例属于旧说明残留，本文未采用。官方v3.2.1同时包含950支持，不改变本文只选择A2/A3的范围。

上游原始指南没有性能前后数字。提供代码对照和待上板benchmark，不声明速度提升。

原始逐文件 hash/URL：[`source-manifest.json`](../code/source-manifest.json)。本地 golden 和NPU wrapper不是上游发布的完整KV allocator。
