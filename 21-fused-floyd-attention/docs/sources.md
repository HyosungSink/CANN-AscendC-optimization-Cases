# 来源与版本证据

核对日期为 2026年10月8日。ops-transformer 正式 `v9.0.0` 固定为 `afe72144f9f2ac8441929035795db88a111b30c5`。[官方发布说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)明确配套 CANN 9.0.0。

- [算子 README](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/fused_floyd_attention/README.md)：A2 训练/A3 产品支持和 shape 约束。
- [设计与性能原表](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/fused_floyd_attention/docs/FFA%E7%AE%97%E5%AD%90%E8%AE%BE%E8%AE%A1%E4%BB%8B%E7%BB%8D.md)：CV tiling、流水、Chunk 对照和显存数据。
- [主流水](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/fused_floyd_attention/op_kernel/fused_floyd_attention_s1s2_bn2gs1.h)：两路 QK/PV、三任务记录和结果双槽。
- [Host Tiling](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/attention/fused_floyd_attention/op_host/fused_floyd_attention_tiling_general.cpp)：nRatio、平台资源与 BatchMatmul 配置。

原性能表时间单位为秒，文章换算成毫秒；显存表沿用 GB。表中未列测量 CANN、驱动、完整 baseline 版本和内存峰值工具，因而成绩被标为上游参考值。正式源码配套版本与历史测量版本分别记录，不互相替代。

固定版本与核心源码链接见本页来源表或 [来源说明](sources.md)，快照保留原始版权和许可证；逐文件 SHA256 见 [整包清单](../../bundle-manifest.json)。
