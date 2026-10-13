# 来源和版本证据

核对日期为 2026年10月8日。上游为 [hw-native-sys pto-isa](https://github.com/hw-native-sys/pto-isa)，快照 commit 为 `15a9e0a0845955f5d7a409a7f4d1609a263b5d25`。所有快照文件的原始 URL 和 SHA256 见 [source-manifest.json](../code/source-manifest.json)。

| 证据 | 固定来源 | 支持的结论 |
| --- | --- | --- |
| 案例 README | [README_zh.md](https://github.com/hw-native-sys/pto-isa/blob/15a9e0a0845955f5d7a409a7f4d1609a263b5d25/kernels/manual/a2a3/gemm_ar/README_zh.md) | A2/A3 示例，910B 八卡性能；变更记录明确兼容 CANN 9.0.0 |
| CMake | [CMakeLists.txt](https://github.com/hw-native-sys/pto-isa/blob/15a9e0a0845955f5d7a409a7f4d1609a263b5d25/kernels/manual/a2a3/gemm_ar/CMakeLists.txt) | A2/A3 架构宏、dav-c220 Cube/Vector targets 和 CANN/HCOMM 依赖 |
| 计算内核 | [gemm_compute_kernel.cpp](https://github.com/hw-native-sys/pto-isa/blob/15a9e0a0845955f5d7a409a7f4d1609a263b5d25/kernels/manual/a2a3/gemm_ar/gemm_compute_kernel.cpp) | stepK、L1/L0 双缓冲和块映射 |
| 通信内核 | [comm_kernel.cpp](https://github.com/hw-native-sys/pto-isa/blob/15a9e0a0845955f5d7a409a7f4d1609a263b5d25/kernels/manual/a2a3/gemm_ar/comm_kernel.cpp) | AtomicAdd、subtile ready、AG drain 和内存同步协议 |
| 测量程序 | [main.cpp](https://github.com/hw-native-sys/pto-isa/blob/15a9e0a0845955f5d7a409a7f4d1609a263b5d25/kernels/manual/a2a3/gemm_ar/main.cpp) | 预热、计时口径和数值验收的实际实现 |
| 许可证 | [LICENSE](https://github.com/hw-native-sys/pto-isa/blob/15a9e0a0845955f5d7a409a7f4d1609a263b5d25/LICENSE) | CANN Open Software License Agreement Version 2.0，保留通知并提供许可证 |

公开耗时来自上游案例 README，其 CANN 测量版本、驱动版本和原始 profiler 日志未在该表列出。文章以兼容声明和 A2/A3 实现作为收录依据，并将成绩标为上游参考值。历史优化表的阶段不是当前源码的独立开关。

本包新增的 wrapper 仅提供固定源码获取、CMake 构建和 mpirun 启动，不修改上游 kernel。与原始 `run.sh` 的区别是省去系统级 IPC 清理，且把下载目录和构建目录显式分离。原始脚本仍保存在快照中。
