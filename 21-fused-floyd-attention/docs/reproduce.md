# FusedFloydAttention 复现步骤

使用 Linux 与 CANN 9.0.0 Toolkit/ops，匹配硬件的驱动和固件。支持 A2 训练系列，以及 A3 训练/推理系列；产品表明确排除 A2 推理系列。固定 ops-transformer commit `afe72144f9f2ac8441929035795db88a111b30c5`，实际 opbase 依赖为 `929902e577bb809a377a6ab17cb7b00a4dfddc1a`。

```bash
# 在本 post 目录执行
export CASE_WORKDIR="$PWD/work/fused_floyd_attention"
bash code/reproduce.sh fetch
source /usr/local/Ascend/cann/set_env.sh
CASE_SOC=ascend910b bash code/reproduce.sh build
# 支持的 A3 配置使用 CASE_SOC=ascend910_93
```

wrapper 检查上级 `../../ops-base` 的优先依赖，构建完整固定工程，只生成包。按上游源码调用指南在隔离 vendor 环境运行 `attention/fused_floyd_attention/examples/test_aclnn_fused_floyd_attention.cpp` 及对应 UT。阅读快照未包含全仓编译系统，不能仅编译其中一个 header 当作完整算子。

输入约束包含 N 为 16 的倍数、M/K 为 128 的倍数，D 为 32/64/128，具体布局与各输入轴的关系以 API 文档为准。数值 golden 必须相加两路打分后执行同一个 Softmax，然后相加两路 PV；测试 mask、不同 D、最后一个 KV 块和有效输出区域。

性能 baseline 需固定 Chunk 大小、dtype、scale、mask 和是否物化全量分数。记录整个拆分链的 device 时间与融合算子 device 时间，另行测峰值显存。公开原表没有完整计时/内存口径，不能只根据 API 名字就认定 baseline 与原文相同。

检查 nRatio、CV 子块数、两份结果 workspace 和流水末尾两轮排空，改变某个 tiling 参数后重新构建并对拍。保持 CANN、驱动、SoC、源码和依赖 commit 不变，记录原始 profiler 数据。本交付没有 NPU 编译、数值或性能复测结果。
