# 复现 mla_prolog_v3：固定源码、构建、调用和测量

## 当前完成到哪里

已获取并核对 ops-transformer `v9.0.0`，提交 `afe72144f9f2ac8441929035795db88a111b30c5`，检查 A2/A3 注册与实际源码路径，保存原始源码并校验 SHA256。**没有在本机构建 NPU 内核、运行 aclnn 数值比对或测量性能**：本机是 macOS，没有 A2/A3。

`code/upstream/` 是阅读快照，保留了源码版权，排除 950 的 `arch35/`；它不是完整编译工程。下面的编译路线使用固定提交的完整工程。只有实际执行成功才可把状态更新为“构建/运行通过”。

## 1. 准备 A2/A3 Linux 环境

安装正式 CANN 9.0.0 Toolkit、匹配的 ops 子包和 Ascend HDK，版本配套见 [官方 9.0.0 说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)。本文不下载或安装这些二进制包。

在自己的安装目录执行 `source` 相应的 `set_env.sh` / `setenv.bash`，检查 `ASCEND_HOME_PATH`、`bisheng`、`g++`、`cmake`、`python3` 和 `npu-smi info`，并从 Toolkit / ops 安装信息核对正式版本。示例脚本要求用户确认后设置 `CANN_VERSION=9.0.0`，它不会把这个环境变量当成已经验证 NPU 数值的证据。

A2 用 `SOC=ascend910b`，A3 用 `SOC=ascend910_93`。本文 wrapper 不接受 `ascend950`；更高版本的 CANN 应另选配套源码重做验证，不能直接混用本工程。

## 2. 校验本地快照，再取固定完整工程

在你选定的工作目录执行，默认中间文件写入当前目录的 `work/04-deepseek-mla-prolog`：

```bash
bash outputs/blog/04-deepseek-mla-prolog/code/reproduce.sh verify
bash outputs/blog/04-deepseek-mla-prolog/code/reproduce.sh fetch
```

`fetch` 只读取源码仓库，固定：

- ops-transformer：`afe72144f9f2ac8441929035795db88a111b30c5`。
- opbase：`929902e577bb809a377a6ab17cb7b00a4dfddc1a`，来自该工程实际的 `cmake/third_party/opbase.cmake`。

已存在的 checkout 若提交不一致，脚本会停止，不会覆盖用户的另一份源码。可通过 `CASE_WORK_DIR=/absolute/work/path` 选择工作目录。

## 3. 准备离线依赖，构建自定义算子包

完整工程的打包流程需要 json、Eigen、makeself、protobuf、abseil 等第三方源码；它们不属于本文 Kernel。按 [官方编译说明](https://gitcode.com/cann/ops-transformer/blob/afe72144f9f2ac8441929035795db88a111b30c5/docs/zh/invocation/quick_op_invocation.md) 准备，放在 `work/04-deepseek-mla-prolog/third_party/pkg/`：

```text
include.zip                             # json 3.11.3
eigen-5.0.0.tar.gz                       # Eigen 5.0.0
makeself-release-2.5.0-patch1.tar.gz
protobuf-25.1.tar.gz
abseil-cpp-20230802.1.tar.gz
```

可用 `THIRD_PARTY_DIR` 指定另一份依赖缓存。wrapper 在这些归档缺失时拒绝启动上游 build，避免自动下载大包；不调用 sudo，不自动安装 CANN 或自定义包。本文未验证第三方包构建成功，实际构建日志应保留。

```bash
CANN_VERSION=9.0.0 SOC=ascend910b BUILD_JOBS=8 \
  bash outputs/blog/04-deepseek-mla-prolog/code/reproduce.sh build
```

A3 改为 `SOC=ascend910_93`。实际调用的上游命令是：

```bash
bash build.sh --pkg --soc=ascend910b --ops=mla_prolog_v3 \
  --vendor_name=blog04 --cann_3rd_lib_path=/absolute/third_party -j8
```

MlaPrologV3 的共享 `mla_prolog` 依赖由上游 Host CMake 的 dependency 声明处理，不要仅拿 V3 入口文件单独编译。构建产物在完整工程的 `build_out/`。本步骤只构建包，不执行安装。

## 4. 在测试环境装入已构建包，运行指定样例

使用上游说明安装已经生成的**自定义算子包**，记录具体 `.run` 文件与安装位置。不要用本文的示例覆盖整套内置 ops-transformer 包。安装与环境挂载的命令以构建产物和上游文档为准；wrapper 不会自动执行。

自定义 vendor 的 `set_env.bash` 把路径加入 `ASCEND_CUSTOM_OPP_PATH` 后，找到其 `op_api` 目录，该目录需包含 `include/` 和 `lib/`。设定：

```bash
export CUSTOM_OPAPI_ROOT=/absolute/path/to/vendors/blog04_transformer/op_api
CANN_VERSION=9.0.0 SOC=ascend910b \
  bash outputs/blog/04-deepseek-mla-prolog/code/reproduce.sh run-example
```

该模式只编译并调用本文指定的一个 A2/A3 示例，动态链接 `libcust_opapi` 以及 CANN ACL/nnopbase/math 库；不同安装布局需核对 include/lib 路径。若自定义 header/library 没被正确加载，运行成功可能实际走到内置算子，必须检查 profiler 的 kernel 与自定义包路径。

**样例限制：BF16 零初始化样例只验证基本 API 流程；随机输入、NZ 权重转换和半量化路径需要另外验证。**

## 5. 从调用成功到可发布的优化结论

先跑数值：生成非退化随机输入，按真实 NZ 布局排权重；对照相同 dtype、scale 和 Cache/groupList 契约的参考实现。记录有效输出范围、最大绝对/相对误差；SwiGLU 和量化场景要比较输出 scale 与量化数据，而非只看单个元素。

再跑性能：预先完成内存分配、权重转换与输入搬运，预热，重复执行并用同步后的 device 事件或官方 profiler 提取 **Device 算子时间**。公开中位数及离散程度；CPU wall-clock 可附带记录，但应与 device 时间分列。

对比融合版本和拆分基线时固定 shape、专家/Cache 分布、数据类型、精度约束与输入数据，并同时记录整个拆分链的总 device 时间。仅改模板开关的实验要记录改动 diff 与新 commit，不把它称为上游已有结果。

推荐结果表：

| SoC / 设备 | Toolkit / ops / HDK | shape / dtype / layout | 基线 | 实现提交 | Device μs | 数值误差 | workspace |
| --- | --- | --- | --- | --- | ---: | --- | ---: |
| 待测 | 待填 | 待填 | 待填 | 待填 | 待测 | 待测 | 待测 |

当前目录没有这些 NPU 实测结果。文章中每个 tile、同步和缓存选择都能在固定原始源码中核对；其实际收益仍须由上述测量闭环证明。
