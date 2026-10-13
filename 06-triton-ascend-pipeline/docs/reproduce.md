# 复现：页面索引生成的四组算子对照

## 环境和版本

目标：Linux x86_64/aarch64，Atlas A2（910B）或 Atlas A3，**CANN 9.0.0 + Triton-Ascend v3.2.1**。固定 commit 为 `2badfc89e70a9b7a5e88463a116c2feddce4b101`；正式 release notes 的配套信息保存于 [`upstream-release.json`](upstream-release.json)。A3 与 A2 是平台范围，不把 A3 的硬件名称统一写成 910B。950 不在本篇复现范围。

运行前的设备名称检查只接受 `Ascend910B` 及其型号后缀、`Ascend910_93xx`、`Ascend910C` 及其型号后缀。310、950、旧或模糊的 `Ascend910`、未知名称一律拒绝；没有绕过开关。若真实 A2/A3 返回其他名称，需先用 npu-smi 核实具体板型和上游支持证据，再明确扩展白名单。

PyTorch、torch-npu、Python、CANN Toolkit/ops、驱动与固件由使用者提前准备并确认配套。上游 release 列举 Python 和 PyTorch 支持版本；以实际 torch-npu/CANN 配套约束选择组合。保存 `npu-smi info`、CANN `version.cfg`、`python3 -m pip show triton-ascend torch torch-npu` 和完整构建 commit。

**注意旧说明残留：**固定源码 `quick_start.md` 的 Docker 示例仍推荐 8.5.0；本篇按 v3.2.1 正式 release 的 CANN 9.0.0 配套，不复制旧 Docker 安装命令。`triton.__version__` 在固定源码仍可能显示底层 Triton 的 3.2.0，不能只靠它判断 Triton-Ascend release；需要 distribution 版本/固定源码构建证据。

## 离线检查

在 post 目录：

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh cpu-check
```

这两项不需要 torch、Triton、CANN 或 NPU。CPU 检查验证 golden 的边界语义和设备名称白名单，不代表 Triton Kernel 在设备上的正确性。当前主机已运行 7 个 golden 检查，以及 15 个设备名称接受/拒绝子用例，全部通过；已检查 Python AST、Shell 语法与原始源码 SHA-256。

## 可选固定源码获取

```bash
bash code/reproduce.sh checkout
```

这是显式联网下载源码动作，默认目录 `code/work/triton-ascend`；不会安装 CANN、PyTorch 或编译器依赖。准备完整匹配环境或自行按上游构建说明安装固定 release；不使用 `pip install -U` 追随 latest 作为复现步骤。若已由固定源码安装，记录它与包之间的对应证据。

## 上板 correctness 和 timing

先 source 已安装 CANN 环境文件。当前 post 目录：

```bash
bash code/reproduce.sh benchmark --output code/work/results.json
```

默认运行 batch=8、page_size=16、TILE=1024、新增 token 上限 128 / 1025 / 4096，随机种子 20261008。每组输出包含旧/新长度、空闲页编号、逻辑上限与编译上限，输入使用 int32。四个变体都必须先逐元素匹配 CPU golden，包括未写入的 `-1` 槽位。

```bash
# 额外探索tile和尾块；参数必须是合法的Triton编译规格
bash code/reproduce.sh benchmark --batch 8 --max-extend 1091 4097 --tile 512 --warmups 10 --repeats 30 --output code/work/tile512.json
bash code/reproduce.sh benchmark --batch 8 --max-extend 1091 4097 --tile 1024 --warmups 10 --repeats 30 --output code/work/tile1024.json
```

输出 `golden=PASS` 后才可解释时间；若 Kernel 编译报错，保留错误，不能把失败规格列为已优化成功。测试不自动设置对所有规格都适用的性能阈值。

## 计时边界

- 每个变体的第一次编译、输入构建、CPU golden、输出检查均不计时。
- 先预热 10 次，再测量 30 次；NPU Event 包住一次 Kernel 调用，在同一 stream 中同步结束事件。
- 输出为 median / mean / min / max 和全部样本，单位微秒。
- 缓冲持续复用，属于暖缓存事件时间；不等同于冷缓存时间或 profiler 纯 Kernel duration。
- multi-buffer True 只表示提交了编译选项，不证明实际重叠。应用 msprof 或查看生成 IR 后再说明流水效果。

上游指南没有公开这些改写的具体性能表。本篇也没有生成 NPU 数据。结果文件只在设备实际运行后产生，不能预填一个假设加速倍数。
