# 复现：算子正确性、INT8/BF16 比较和设备计时

本篇源码来自 `653e519cf9556b0cc8d9e7966aa4551eb65143ae`。本机仅完成源码完整性和脚本静态检查；下列 NPU 构建/测试命令尚未在设备运行。

## 1. 环境

- Linux x86_64/aarch64，Atlas A2（910B）或 Atlas A3；不选择 A5/950。
- CANN 社区版 **9.0.0 或 9.1.0**，安装对应 Toolkit/ops、驱动、固件。
- PyTorch/torch-npu 至少满足上游配套要求；固定 CI 9.0.0 使用 Python 3.11，9.1.0 使用 Python 3.12。不要在已有环境自动升级。
- 至少 8 个可用 NPU、HCCL 拓扑连接正常。单卡不能复现 EP8。
- A2 专属文档与通用文档对驱动下限写法不同；按 A2 专属文档与所选 CANN 的实际配套表准备，不把一个通用下限当作所有硬件可互换的承诺。

首先保存环境证据：`npu-smi info`、CANN `version.cfg`、`python3 -m pip show torch torch-npu deep-ep`、上游 commit、固件、EP 拓扑和环境变量。A3 不应记录为 910B；A2 才是本案例的 910B 路径。

## 2. 获取完整固定源码

在本 post 目录运行：

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh checkout
cd code/work/sgl-kernel-npu
git submodule update --init --recursive
```

最后一行由使用者明确运行，下载固定子模块；它不包含在自动脚本中。此处快照未包含完整构建依赖。`SOURCE_DIR=/path/to/pinned/sgl-kernel-npu` 可以改用已经准备好的源码。

## 3. 构建

先 source 已安装 CANN 的 `set_env.sh`。回到 post 目录：

```bash
# A2 与 A3 二选一
HARDWARE=A2 bash code/reproduce.sh build
HARDWARE=A3 bash code/reproduce.sh build
```

产物在完整上游的 `output/`。如果选择安装 wheel，应在自己的隔离 Python 环境中手动执行安装，并参考上游说明处理 `deep_ep_cpp` 的导入路径。脚本不会运行 pip，也不会下载/安装 CANN 或模型。已安装包必须来自同一个固定 commit；仅源码 HEAD 匹配不能证明正在导入的 wheel 匹配。

## 4. 上板测试规格

```bash
HARDWARE=A2 bash code/reproduce.sh test-int8
HARDWARE=A2 bash code/reproduce.sh test-bf16
# 在 A3 设备把 HARDWARE 改为 A3
```

脚本规格：EP8、每 rank 128 tokens、hidden=7168、topk=8、专家数=256、策略=`default`。INT8 模式使用上游 `--use-fp8` 参数；它在 A2/A3 的真实含义是 INT8 Dispatch。BF16 模式不传该参数。Combine 输入由测试把接收 payload 反量化回来，不包含专家 FFN 的实际计算。

`NUM_PROCESSES=16/32` 可改变 EP，但它要求对应的可用设备及拓扑，不是只改数字就能在 8 卡单机上成功。公开 A3 表未注明专家数；本脚本的 256 是上游测试默认和明确选择的复现规格，不能声称它完整恢复了历史表的全部条件。

先检查所有 rank 的正确性输出和无 NaN，再比较 profiler 单算子时间。测试还支持：

```bash
cd code/work/sgl-kernel-npu
python3 tests/python/deepep/test_low_latency.py --num-processes 8 --num-tokens 128 --hidden 7168 --num-experts 256 --num-topk 8 --use-fp8 --enable-dynamic-tokens --drop-percent 0.1
```

这一项用于覆盖各 rank token 数不一致与无效专家 `-1`，不与固定 128-token 的性能结果混合。不要默认运行 `--pressure-test`：它会进行大量循环。

## 5. 保存结果与解释

保存 `PASSED` 精度输出、每 rank 的单独 Dispatch/Combine duration、平均/最小/最大值和环境信息。测试的 NPU Event 配对计时包含测试反量化；Profiler 设备事件是另一口径。公开 132/126 μs 的 CANN、驱动与当时 commit 未单列，本文不设置它们为 pass/fail 阈值。

在同一板型同一拓扑的 INT8/BF16 对照中，仅比较 Dispatch 是否收益、是否满足量化误差；这也不代表 FFN+Combine 的整网收益。需要从 profiler 确认实际进入 A2 或 A3 Kernel，再讨论差异。

A2 多节点测试需要按保存的 `A2_DEEPEP.md` 设置网络环境并手动验证。固定快照已关闭 A2 多节点 CI；本文的脚本只给出单节点起点，未声称已复现多节点性能。
