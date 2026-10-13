# BatchMatmulMaxSum · Ascend C 优化项目

本项目实现 CANN 挑战赛的 BatchMatmulMaxSum 算子：对每个 batch 计算 `BMM(X1, X2)`，沿 `N` 取最大值，再沿 `M` 求和，输出 FP32 分数。支持 FP16/BF16 输入及两路输入各自的转置存储。完整约束见 [赛题说明](race_intro.md)。

当前提交版本已在远端 Ascend 910C 上通过 15 个已知形状的回归测试，并在 [OJ 提交](https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum/submission/6ab4fdfa694b590c3c16f181) 中通过 15/15；case 8 用时 85.11 μs。这是当次测量，不代表后续修改仍然通过。

## 从哪里开始

| 文件 | 用途 |
|---|---|
| `src/kernel.asc` | 唯一需要提交到 OJ 的 Ascend C 源码；入口为 `run_kernel` |
| `src/debug_main.asc` | 接受 `B M N K dtype transA transB` 参数的本地测试程序，生成输入并检查结果 |
| `scripts/remote_test.sh` | 从本机复制源码到远端临时目录、构建并运行测试 |
| `src/scripts/oj_cases.sh` | 15 个已知形状的正确性与性能回归 |
| `src/scripts/representative_cases.sh` | 覆盖小矩阵、长 K、宽 N、尾块和转置的代表形状 |
| `doc/case.md` / `doc/optimal_log.md` | 已知 case 参数与每轮优化、测量和撤回记录 |

先读 `doc/race_intro.md` 明确数学语义，再读 `src/kernel.asc` 和 `doc/optimal_log.md`。`doc/case.md` 中的形状用于复现测试；测试输入由本项目生成，并非 OJ 的隐藏输入。

## 运行环境

构建和执行需要 Ascend NPU、CANN 工具链、CMake、C++ 编译器及 Python 3 + NumPy。性能采集额外需要 `msprof`。当前远端测试环境为 Ascend 910C / CANN 9.0.0；`src/CMakeLists.txt` 默认编译目标为 `dav-2201`。本机没有 CANN 时，可在本地改代码，通过 SSH 在远端构建测试。

`scripts/remote_test.sh` 依赖本机的 `scripts/ssh_npu.sh`。这个连接脚本不在 Git 仓库中；为自己的服务器创建它，接口是“接收一个远端 shell 命令，执行并透传标准输入/输出”。最简单的内容是：

```bash
#!/usr/bin/env bash
exec ssh "$NPU_HOST" "$@"
```

运行前设置 `NPU_HOST=user@server`，并先确认 `bash scripts/ssh_npu.sh 'npu-smi info'` 能成功。不要把密钥、密码或含凭据的连接脚本提交到仓库。`remote_test.sh` 目前将远端 CANN 路径固定为 `/usr/local/Ascend/ascend-toolkit/latest`；服务器路径不同时先调整该脚本。

## 常用命令

在仓库根目录运行：

```bash
bash scripts/remote_test.sh                    # 构建并跑基础样例
bash scripts/remote_test.sh --representative   # 代表形状正确性测试
bash scripts/remote_test.sh --stress           # 代表形状 + 上界压力测试
bash scripts/remote_test.sh --oj-check         # 15 个已知 case 的正确性测试
bash scripts/remote_test.sh --oj-check 8       # 只检查 case 8
bash scripts/remote_test.sh --oj-profile 8     # 检查 case 8 并采集单次 kernel 用时
bash scripts/remote_test.sh --oj-profile       # 全部 15 个 case 的性能采集
```

每次运行会在远端创建独立的 `/tmp/cann-race.*` 目录，并在其中执行 `src/run.sh`。若直接在 NPU 机器上工作，可运行 `ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest bash src/run.sh`。`run.sh` 会重建 `src/build/`；不要在该目录保存手工文件。

`--oj-profile` 输出的 `kernel_us` 来自 `msprof` 的单次 NPU task，不含主机初始化，但也**不等于 OJ 用时**；只在同一环境中比较优化前后版本。`--oj-check` 校验数值，不采集性能。`doc/case.md` 记录了 15 个测试参数。

## 给接手的 agent

1. 只改必要文件，保留工作区中已有的未提交文件；特别不要提交私有的 `scripts/ssh_npu.sh`。
2. 修改 `src/kernel.asc` 后，至少运行相关 case 和 `--oj-check`；改动通用路径时再跑 `--representative`。OJ 要求每次调用恰好启动一个 kernel，不能用多次 launch 拼接结果。
3. 优化时同时记录正确性、`kernel_us`、对照版本和撤回原因到 `doc/optimal_log.md`。通过远端测试后，才能将 OJ 测量归因于本次改动；不要把两种用时混用。
4. 需要真实 OJ 验证时，提交的是 `src/kernel.asc` 的完整内容；提交前后核对网页保存的源码与本地文件一致。OJ 页面为[赛题入口](https://cannjudge.cn/public/op_challenge_shanghe_prelim/batchmatmulmaxsum)。
