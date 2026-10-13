# SparseFlashAttention 的固定版本复现

目标是复现 CANN 9.0.0 正式配套的单算子 Kernel，并进行输出检查和计时。本机为 macOS，未进行 CANN 编译、NPU 执行或性能测量。本文中的命令依据固定版本的上游 [算子调用指南](../code/upstream/docs/zh/invocation/quick_op_invocation.md)；成功状态需由目标机器实际结果确认。

## 环境和版本

主复现为 Atlas A2 推理系列 / Ascend 910B。虽然 README 另列 A3 推理支持，本提交 Host 的 GetNpuInfo 只接受 ASCEND910B，A3 源码复现需另行核对上游修正，脚本不接受 ascend910_93。

完整工程固定为：

```text
repository  https://gitcode.com/cann/ops-transformer.git
ref         v9.0.0
commit      afe72144f9f2ac8441929035795db88a111b30c5
opbase      929902e577bb809a377a6ab17cb7b00a4dfddc1a
```

在 Linux NPU 机器预先准备 CANN 9.0.0 toolkit、对应 SoC 的算子软件包及匹配驱动/固件，按安装目录中实际的 `set_env.sh` 或 `bin/setenv.bash` 生效环境，确认 `ASCEND_HOME_PATH`。构建需要上游要求的 CMake、编译器及 Python 依赖。本文不自动安装软件包。

记录 CANN 安装元数据中的完整版本、驱动/固件版本、`npu-smi info` 的设备型号和状态。A2 的构建目标是 `ascend910b`，A3 的目标名是 `ascend910_93`，不能把 A3 简写成 910B。9.0.0 的正式配套表允许相应的 HDK 版本，见 [本地配套证据](evidence/cann-9.0.0-release-notes.md)。若使用 CANN 9.1.0 或更新版本，先根据对应正式 release notes 重新选择配套源码标签；本文不会自动外推 ABI/编译兼容性。

## 获取完整源码与构建

在本 post 根目录执行：

```bash
export BLOG_WORK_DIR=/path/to/scratch/sparse_flash_attention
bash code/reproduce.sh fetch ascend910b
bash code/reproduce.sh build ascend910b
```

fetch 将正式标签 checkout 固定为完整提交并核验 HEAD。若目录已有其他提交或未提交改动，会停止，避免覆盖工作。完整源码所需的其他目录由 git 一并取得；`code/upstream/` 只供阅读，不能作为顶层构建工程。

build 对应的实际命令是：

```bash
bash build.sh --pkg --soc=ascend910b --ops=sparse_flash_attention --vendor_name=blog_sfa
```

上游 CMake 可能下载第三方依赖，opbase 的提交由 [配置](../code/upstream/cmake/third_party/opbase.cmake) 固定。离线构建时，按上游表格准备第三方包，设置 `BLOG_THIRD_PARTY_DIR=/path/to/third_party`。此脚本不下载 CANN 安装包，也不执行上游 `install_deps.sh`。

完成后检查 `build_out` 内实际生成的自定义算子 `.run` 包。安装是一个单独、手动的步骤，选定实际包名后查看它的 `--help`，按本机器的可写 vendor 目录安装。随后按生成包提供的 vendor `set_env.bash` 或安装规则配置 `ASCEND_CUSTOM_OPP_PATH`。不要用全量 ops-transformer 包覆盖现有环境来做单算子实验。

自定义包和系统 CANN 算子共存时，必须确认自定义包的加载优先级；否则性能变化可能仍来自系统里的旧内核。示例辅助命令：

```bash
bash code/reproduce.sh example ascend910b
bash code/reproduce.sh ut-host ascend910b
bash code/reproduce.sh ut-api ascend910b
```

example 走上游 `test_aclnn_sparse_flash_attention.cpp` 的 `cust` 路线。Host/API UT 验证解析、Tiling 或 API 包装的逻辑；这些 UT 通过不等于 Kernel 数值与性能已经通过。

## A3 的证据边界

产品 README 列出 A3 推理支持，但本提交 `SFAInfoParser::GetNpuInfo` 只接受 `ASCEND910B`。本复现脚本限制为 A2/910B，不能通过把 `--soc` 改成 A3 来宣称支持成立。若要做 A3，需先取得上游对 Host 检查与设备信息映射的说明或已修正提交，再记录新的版本配套和性能，不能悄悄改检查后仍标注本固定提交。

## 单算子精度与计时

上游参数示例的典型规格为 `B=4, S1=1, S2=8192, N1=128, D_nope=512, D_rope=64, sparse_count=2048, sparse_block_size=1`。本目录另外提供可运行的 PyTorch 辅助脚本，需要与本 CANN 版本配套、且确实导出 `torch_npu.npu_sparse_flash_attention` 的 torch/torch_npu。是否存在 Python API 由脚本运行时检查；若缺少接口，使用上游 aclnn example 路线，不能用别的算子名字替代。

```bash
python3 code/benchmark.py --cann-version 9.0.0 --batch 4 --s1 1 --s2 8192 --heads 128 --topk 2048 --block-size 1 --dtype fp16
```

脚本使用已量化到输入 dtype 的 CPU 数据构造 FP32 参考，Kernel 在 NPU 上运行，参考计算不包含在计时区间。只覆盖 BSND、完整有效长度、无 causal mask；它不覆盖 TND、PageAttention、实际长度尾部以及返回 LSE/score 的全部接口组合。`--cann-version` 是用户根据安装元数据提供的记录值，不是脚本独立探测证明，输出会注明此点。

通过 warmup 后逐次调用并同步，用 `perf_counter` 记录 p50/p90 的单算子调用耗时，包含 Python 提交和同步开销。它适合对比相同 harness 的改动，不能直接等同于设备 Kernel 执行时间。需要纯设备时间时使用 msProf，保留 profiler 记录和同一套输入，参照 [上游调优指南](../code/upstream/docs/zh/debug/op_debug_prof.md)。

正确性需要区别不同情况：TopK 同分允许不同索引次序；Attention 低精度 MM2 与在线重标度可能使输出与全 FP32 参考有数值误差。脚本打印实际误差并按显式容差检查。若失败，先排查输入条件与接口/包版本，再扩大覆盖或调整有依据的容差，不能只删掉检查。

## 消融和结果记录

[正文](post.md) 给出具体源码分支的消融。请为原版和每个改动分别创建独立完整工程与 vendor 名，保存 diff、固定提交、安装记录、正确性结果和 profiler；不要把改过的工程交给拒绝 dirty checkout 的 `reproduce.sh`，手工按同一 build 命令编译更明确。

每条结果应保存 CANN、HDK、设备、B/S1/S2/head、输入和输出精度、稀疏数量/块大小、索引分布、layout、mask、有效长度、warmup 和重复次数、计时口径。报告中没有测试过的格子保留未测状态；此交付物没有 NPU 测量结果。

## opbase 来源优先级保护

上游 CMake 优先读取完整工程上两级的 `../../ops-base`，其次才是指定第三方缓存中的 `opbase`。仅传 `BLOG_THIRD_PARTY_DIR` 不能覆盖前者。复现脚本在构建/UT/示例动作前检查这两处已存在的候选：必须是独立 Git checkout、HEAD 为本版本规定的 opbase commit，且没有未提交或未跟踪文件；否则停止并要求使用隔离工作目录。依赖不存在时仍由固定上游 CMake 获取，脚本不会删除或改写其他工作目录的依赖。
