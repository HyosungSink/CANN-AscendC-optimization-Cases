# GEMM AllReduce 复现步骤

## 环境和版本

- Linux，Ascend A2 或 A3，已经安装匹配的 NPU 驱动和固件。公开性能参考平台为 8 卡 Ascend 910B。
- CANN Toolkit 9.0.0 或更高版本，使用该 Toolkit 的 Bisheng、ACL、HCCL 和 HCOMM。上游明确记录了 9.0.0 兼容改动，未给出所有后续版本逐一验证表。
- CMake 3.16 或更高、MPICH、可被示例动态加载的 `libmpi.so`。按本示例 CMake 的 C++17 配置构建。
- PTO 固定 commit `15a9e0a0845955f5d7a409a7f4d1609a263b5d25`。

以下命令在本 post 目录执行。选择空的工作目录，源码和构建产物会放在那里。

```bash
export CASE_WORKDIR="$PWD/work/pto-gemm-ar"
bash code/reproduce.sh fetch

# 按实际 CANN 安装位置填写，不要把不同版本的 Toolkit 混在 PATH 中。
export ASCEND_CANN_PATH=/usr/local/Ascend/cann/set_env.sh
export ASCEND_DRIVER_PATH=/usr/local/Ascend/driver
export PATH=/usr/local/mpich/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/mpich/lib:${LD_LIBRARY_PATH:-}
export MPI_LIB_PATH=/usr/local/mpich/lib/libmpi.so

bash code/reproduce.sh build
CASE_NRANKS=8 CASE_FIRST_DEVICE=0 bash code/reproduce.sh run
```

本地 wrapper 不执行上游 `run.sh` 中的 `ipcrm -a`。本快照可以离线阅读；完整 NPU 编译和数值、性能验证需要上述 Ascend 环境，当前交付不包含新的上板测量结果。

## 参考输入

保持 `M=5416, K=6144, N=1408`、24 个计算 block 和 24 个通信 block，检查日志中的 padded shape 为 `5504×6144×1536`、tile 数 258。默认 K 要能被 `64×4=256` 整除。若改变维度，重新按 pad 后输出大小设置足够的 `HCCL_BUFFSIZE`，所有 rank 应保持一致；200 MB 适用于这里的默认输出规模。

## 记录计时和数值结果

源码输出 Compute only、Sequential、Pipelined 及各阶段统计。同一进程同时覆盖串行和并发路径；二者分别预热 5 次，测量 10 次。再次测量时保持芯片型号、频率、驱动、CANN、MPICH、rank 和内存布局一致。记录全量原始日志，不只保存最快的一次。

原始精度检查在 rank 0 对额外执行的串行输出进行验证，使用较宽的容差并允许一定异常元素比例。确认性能收益之前，另行校验每个 rank 的并发路径，至少检查原始有效区域、padding 边界、不同矩阵尺寸以及最坏误差。CPU 仿真不能证明多卡同步和内存可见性正确。

## 建议的单变量实验

1. 固定默认 shape 和计算 block 数，将通信 block 数设置为 12、24，再观察 Host 总耗时。若要尝试 48，先确认设备核数和源码最大值；历史记录显示更多 AIV 可能降低性能。
2. 固定核数，调整 tile 或 stepK，先复查 L0 和 L1 容量与对齐，再比较计算与通信等待。该快照的公开 wrapper 只直接支持 shape 和 block 参数，tile 修改应在独立副本中完成并记录 diff。
3. 保持每个实验的精度标准一致。历史优化表不能作为当前开关消融结果，各阶段代码的独立收益需要对应历史版本。
