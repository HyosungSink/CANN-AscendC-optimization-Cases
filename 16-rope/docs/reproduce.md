# 复现 RoPE 源码与上板验证

## 固定环境

Linux、Atlas A2（910B）或 Atlas A3，CANN 9.0.0 Toolkit/ops 和匹配驱动/固件。源码：ops-transformer 正式 v9.0.0，commit `afe72144f9f2ac8441929035795db88a111b30c5`。版本配套证据保存在 `evidence/release-notes-9.0.0.md`；硬件范围以算子 README 和 A2/A3 host config 为准。

先准备已安装的 bisheng/CMake、opbase 等开发依赖并 source CANN `set_env.sh`。记录 CANN `version.cfg`、npu-smi、完整源码 commit、安装包信息和实际加载的算子路径。脚本不会安装/升级依赖。

## 离线检查与固定源码获取

在 post 目录：

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh checkout
```

默认源码工作目录 `code/work/ops-transformer`；`SOURCE_DIR=/path/to/fixed/repo` 可以使用已准备的同一 commit。快照是阅读文件集，需要完整上游构建树。

## 构建和接口运行

```bash
SOC_VERSION=ascend910b bash code/reproduce.sh build
# A3使用 ascend910_93
```

产物位置以完整上游 build 输出为准。先在目标隔离开发环境手动安装所构建的自定义算子包、按包说明 source 环境，再运行：

```bash
SOC_VERSION=ascend910b bash code/reproduce.sh run
```

wrapper 调用固定项目的 `--run_example rotary_position_embedding eager cust --vendor_name=blog_rope`。默认样例是 float、shape `{1,1,1,128}`、mode=1，只是接口冒烟，不能作为性能样例。安装包和源码 commit 的匹配需要独立记录；脚本检查 HEAD 不代表系统已安装同一版本。

## 具体调优验证规格

在工作 checkout 的样例中同步修改 X/cos/sin/out shape 和数据长度，保留原样例副本及修改 diff：

| 目标分支 | X规格 | cos/sin规格 | 模式 |
|---|---|---|---|
| BNSD广播 | `[2,8,257,128]` | `[1,1,257,128]` | 0 / 1 |
| 非对齐D | `[2,8,257,126]` | `[1,1,257,126]` | 0 / 1 |
| B1SD half复用 | `[2,8,257,128]` | `[2,1,257,128]` | 0 |

新规格属于建议验证集，尚未在 NPU 编译或执行。FP16/BF16 的 Host 数据需要相应 dtype 构建，不要只改 ACL dtype 而保留 float bytes。使用模式对应的 FP32 公式做 golden，检查输出误差、尾部与跨步数据，不仅观察程序返回值。

## 性能记录

先确保正确性，再使用已安装的 msprof 包住编译生成的示例可执行文件，查看 `RotaryPositionEmbedding` 设备事件。样例原本包含 ACL 初始化、H2D、D2H、打印和释放；shell 总耗时不能作为 Kernel duration。若需要对照拆分公式，固定同一板型、dtype、布局、广播与shape，并记录重复样本。

当前只有文件 hash 与 Shell 静态检查；本机无 CANN/NPU，未编译自定义包、未安装包、未做设备正确性或性能测量。没有设定虚构的加速阈值。
