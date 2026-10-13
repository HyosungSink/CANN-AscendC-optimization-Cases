# RotaryPosEmb（RoPE 旋转位置编码）算子工程

Ascend C Kernel 直调工程（`.asc` 直调变体）：`main.asc`（host runner）+ `kernel.asc`（算子 kernel），经 cmake + make 构建为本地可执行 `build/rotary_pos_emb_custom`。

> 接口、参数、约束、错误码与调用示例详见 [doc/RotaryPosEmb.md](RotaryPosEmb.md)，本 README 只做工程门面指引。

## 简介

RotaryPosEmb 实现旋转位置编码（Rotary Position Embedding，RoPE）：将 token 向量最后一维（rope 子维度）按"频率 × 位置"的旋转角两两配对旋转，一步完成 **inv_freq 逆频率生成 → cos/sin 三角函数计算（算子内部完成，不作为输入）→ rotate-half 旋转**，在不引入可学习参数的前提下为 Llama、Qwen、DeepSeek、GLM 等主流大模型注入相对位置信息。

## 环境要求

| 项目 | 要求 |
|------|------|
| 芯片 | Ascend910_9362（NpuArch 2201 / DAV_2201 / dav-c220-vec） |
| CANN | 9.0.0（需先 `source <CANN 路径>/set_env.sh`，依赖 `ASCEND_HOME_PATH`） |
| Python | 3.11+（数据生成 / 校验 / 测试 / 性能采集） |
| torch | golden 参考计算依赖（精度对拍基准） |

## 目录结构

```text
rotaryposemb/
├── kernel.asc        # 算子 kernel（SIMD/MemBase：pair 算术 inv_freq bit-exact +
│                     #   Cody-Waite 预缩减 + 快/慢双路径 + RowGroup 行组 tile）
├── main.asc          # 本地 CLI runner（参数解析 / L2 校验 / 统一退出路径）
├── CMakeLists.txt    # cmake 构建定义（默认 dav-2201，NPU_ARCH 可覆盖）
├── run.sh            # 一键构建 + 生成 case0 数据 + 运行验证
├── .gitignore        # 隔离 build/ 与测试/性能运行产物
├── doc/
│   ├── RotaryPosEmb.md   # 算子详细文档（接口/参数/约束/错误码/示例/性能）
│   └── describe.md       # 赛题描述
├── perf/
│   ├── adapter.sh        # 共享 perf 适配层（build / verify / collect 三职责）
│   └── cases.json        # 性能 case 定义（quick/full suite，7 case）
├── scripts/
│   ├── RotaryPosEmb.py   # golden 参考实现（numpy，位级对拍基准）
│   ├── gen_data.py       # case0 测试数据生成
│   └── verify_result.py  # 精度校验（atol/rtol + matched_ratio 双条件）
└── tests/
    ├── run_tests.sh            # 分级功能测试入口（68 条用例）
    ├── cases_*.json            # L0/L1/L2/白盒用例定义
    ├── perf_collect.py         # 性能采集（多轮统计 + 瓶颈维度分解）
    ├── whitebox.md             # 白盒用例说明
    └── ...                     # L2 runner / 特殊值 / 白盒 tiling 检查脚本
```

## 快速开始

环境准备（任一入口都需要）：

```bash
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0   # 按实际 CANN 路径
source "${ASCEND_HOME_PATH}/set_env.sh"
```

**构建 + 运行**（source CANN 环境 → cmake + make → 生成 case0 数据 → 运行验证）：

```bash
bash run.sh        # 内置 case0 (1,7,1,16) float32 theta=10000，输出 "=== PASSED ===" 即通过
```

**全量测试**（68 条用例：L0 冒烟 / L1 评测集 / L2 runner 异常契约 / 白盒）：

```bash
bash tests/run_tests.sh --level L0      # 冒烟（评测集 debug case）
bash tests/run_tests.sh --level L1      # 评测集全量 7 case（权威门禁）
bash tests/run_tests.sh --level L2      # runner 异常契约
bash tests/run_tests.sh --list          # 列出用例
# 其他参数：--case ID（单跑）、--path A|B|C|all、--no-build；环境变量 PERF_DEVICE 选卡
```

**性能采集**（多轮 `--rounds` 统计 + 瓶颈维度分解，默认先 verify 再采集）：

```bash
python3 tests/perf_collect.py                       # full suite，3 轮
python3 tests/perf_collect.py --suite quick         # quick 2 case
python3 tests/perf_collect.py --rounds 5 --label p8 # 增大轮数抑制波动
```

> 注意：仓库根目录共享 `scripts/perf.py` 的 baseline record/compare 工作流按朴素子串匹配 kernel 名，对本算子（`rotaryposemb` vs `rotary_pos_emb_custom`）不可用；性能对比一律以本工程 `tests/perf_collect.py`（归一化匹配）为准。

**线上平台提交说明**：仅提交 `kernel.asc` 单文件（线上平台禁止调试输出等违规词已清理，kernel 错误路径已做静默处理，详见仓内适配记录）。

## 测试与精度

- 用例规模：68 条（L0 / L1 / L2 / 白盒），入口 `tests/run_tests.sh`；
- 精度口径：逐元素 `abs(actual − golden) ≤ atol + rtol · abs(golden)`，整体双条件 **matched_ratio ≥ 0.99 且 max_abs_error 达标**；

| 数据类型 | atol | rtol |
|----------|------|------|
| float32 | 9.77e-4 | 1.53e-5 |
| float16 | 1.95e-3 | 1.95e-3 |

- 权威门禁：`bash perf/adapter.sh verify`（单 case 按 `PERF_CASE_FILE` 注入）与 `tests/run_tests.sh --level L1`。

## 性能结论

终态实测（msprof op，3 轮中位数，10 warm-up + 50 measured，Ascend910_9362）：

| Case | Shape | dtype | median |
|------|-------|-------|--------|
| deepseek_v3 | (2,512,128,64) | float16 | 93.8 µs |
| long_seq | (4,8192,128,64) | float16 | 1877 µs |
| large_batch | (128,8192,8,64) | float16 | 5206 µs |
| large_head_dim | (2,512,256,1024) | float16 | 1783 µs |

7 case 合计相对首轮基线 **-75.1%**（性能迭代 P1/P4/P4b/P7 累计），后续版本性能回归以 `perf_p7b_*` 报告为准入基线；完整表与瓶颈分析见 [doc/RotaryPosEmb.md](RotaryPosEmb.md)「附：性能参考」。

## 详细文档

接口原型、参数说明、支持数据类型、约束、错误码、调用示例与完整性能数据：**[doc/RotaryPosEmb.md](RotaryPosEmb.md)**。
