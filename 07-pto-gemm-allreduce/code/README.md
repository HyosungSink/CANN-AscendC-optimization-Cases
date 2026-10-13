# GEMM AllReduce 源码快照

`upstream/` 保存 PTO 仓库 commit `15a9e0a0845955f5d7a409a7f4d1609a263b5d25` 的完整 `kernels/manual/a2a3/gemm_ar` 示例，以及原始 LICENSE 和版本兼容说明。原始文件未改写。`source-manifest.json` 提供逐文件 URL 和 SHA256。

重点文件是 `gemm_compute_kernel.cpp`、`comm_kernel.cpp`、`ready_queue.hpp` 和 `main.cpp`。这一快照用于阅读；编译还需要全仓 `include/pto/`、CANN Toolkit、NPU 驱动以及 MPICH。`reproduce.sh` 会在指定工作目录获取固定上游完整工程，然后编译同一示例。

```bash
bash reproduce.sh --help
CASE_WORKDIR="$PWD/work/pto-gemm-ar" bash reproduce.sh fetch
# 在 Linux A2/A3 环境配置好 CANN 后：
ASCEND_CANN_PATH=/usr/local/Ascend/cann/set_env.sh \
CASE_WORKDIR="$PWD/work/pto-gemm-ar" bash reproduce.sh build
```

上游 `run.sh` 包含全局 IPC 清理命令。这里的 `reproduce.sh` 使用 CMake 和 mpirun 直接运行，不执行那段清理。示例需要可用的多卡通信环境，公开参考结果来自 8 卡 910B。

代码受上游 [CANN Open Software License Agreement Version 2.0](upstream/LICENSE)约束，版权与许可证已保留。
