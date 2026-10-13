# RmsNorm 的固定版本复现

复现目标为 Linux 上的 Atlas A2/Ascend 910B 或 Atlas A3/Ascend 910_93 单算子。使用 CANN 9.0.0 正式配套源码；本交付主机为 macOS，没有 CANN 编译、NPU 执行或性能实测。命令依据固定工程的 [调用指南](../code/upstream/docs/zh/invocation/quick_op_invocation.md)。

## 源码与软件环境

```text
repository  https://gitcode.com/cann/ops-nn.git
ref         v9.0.0
commit      0413e303e6cd1986b592c3e039f2579ac0f10b62
opbase      ec88904b611b04f9b23b529aee66c78bb7cf6140
operator    rms_norm
```

预先准备 CANN 9.0.0 toolkit、目标设备算子包、匹配驱动/固件、编译器/CMake/Python 和上游依赖。按安装目录实际的 set_env.sh 或 bin/setenv.bash 生效环境，确认 ASCEND_HOME_PATH。记录安装元数据中的完整 CANN 版本、驱动与固件版本，以及 npu-smi info 的设备；源码标签配套表在 [本地证据](evidence/cann-9.0.0-release-notes.md)。CANN 9.1.0 或以上应重新按正式配套表选择相应标签，本文不自动外推兼容性。

## 完整工程构建

在本 post 根目录执行：

```bash
export BLOG_WORK_DIR=/path/to/scratch/rms_norm
bash code/reproduce.sh fetch ascend910b
bash code/reproduce.sh build ascend910b
```

A3 将两处目标改为 ascend910_93。脚本核对完整 HEAD，已有目录的提交不匹配或有未提交改动会停止，不覆盖本地工作。源码快照不是独立工程，构建使用 fetch 后的完整仓库。

实际单算子命令：

```bash
bash build.sh --pkg --soc=ascend910b --ops=rms_norm --vendor_name=blog_rms_norm
```

CMake 可能下载上游第三方依赖，其 opbase commit 已固定。离线构建按上游依赖表预先准备，再设置 BLOG_THIRD_PARTY_DIR=/path/to/third_party。脚本不执行 sudo、install_deps.sh 或 CANN 下载/安装，不自动安装生成 .run 包。

build_out 中出现生成包后，选定实际包名，先看 --help，再按可写自定义 vendor 目录手工安装与生效环境；确保 ASCEND_CUSTOM_OPP_PATH 和加载优先级使用的是这个 vendor，避免把系统旧 Kernel 当作当前快照。不要覆盖整套系统算子包来做消融。

## 示例、测试与性能

```bash
bash code/reproduce.sh example ascend910b
bash code/reproduce.sh ut-host ascend910b
```

example 使用上游 test_aclnn_rms_norm.cpp。该目录若有多个 example，可在完整工程按调用指南使用 --example_name 选择。Host UT 检查 InferShape/Tiling 等行为；如果目录提供 API UT，再执行 `bash code/reproduce.sh ut-api ascend910b`。UT 通过不能代替实际 Kernel 输出检查和 NPU 计时。

精度检查至少覆盖：单算子 y 与 FP32 rstd；按输入 dtype 的中间精度行为构造参考。 使用 [正文](post.md) 的形状边界和消融点，先确认 Tiling 日志/key 与预期匹配。每次只改一个已存在的能力条件或路径选择，原版与改版分别保存独立 checkout、vendor 和 diff。dirty checkout 的保护脚本不适用于已改动的工程，消融时手工按相同 build 命令构建。

计时将数据生成与 H2D 搬运移出区间，warmup 至稳定后重复测量，保留设备同步或 profiler 口径。msProf 使用方法见 [上游调优指南](../code/upstream/docs/zh/debug/op_debug_prof.md)。避免将 Python 提交+同步时间、纯设备 Kernel 时间和多算子 baseline 时间混成一列。

结果至少记录 CANN/HDK、SoC、shape、dtype与参数dtype、epsilon、可选项与输出、Tiling key、UB配置、warmup、重复次数、p50/p90、精度容差和误差。当前没有保存任何 NPU benchmark 数字，A/B 表须由目标硬件实际复测填写。

## opbase 来源优先级保护

上游 CMake 优先读取完整工程上两级的 `../../ops-base`，其次才是指定第三方缓存中的 `opbase`。仅传 `BLOG_THIRD_PARTY_DIR` 不能覆盖前者。复现脚本在构建/UT/示例动作前检查这两处已存在的候选：必须是独立 Git checkout、HEAD 为本版本规定的 opbase commit，且没有未提交或未跟踪文件；否则停止并要求使用隔离工作目录。依赖不存在时仍由固定上游 CMake 获取，脚本不会删除或改写其他工作目录的依赖。
