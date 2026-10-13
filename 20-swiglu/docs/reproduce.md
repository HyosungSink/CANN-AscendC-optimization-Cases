# SwiGLU 复现步骤

使用 Linux、A2/A3、匹配的驱动与 CANN 9.0.0 Toolkit/ops。固定 ops-nn commit `0413e303e6cd1986b592c3e039f2579ac0f10b62`，上游实际 opbase 依赖为 `ec88904b611b04f9b23b529aee66c78bb7cf6140`。更高 CANN 版本应重新选择和验证其配套源码，不能默认混编。

在本 post 目录执行：

```bash
export CASE_WORKDIR="$PWD/work/swi_glu"
bash code/reproduce.sh fetch
source /usr/local/Ascend/cann/set_env.sh
CASE_SOC=ascend910b bash code/reproduce.sh build
# A3 对应 CASE_SOC=ascend910_93
```

脚本从完整固定工程构建，只生成算子包，不执行安装。CMake 优先查找上级 `../../ops-base`；脚本会拒绝该目录中的不同 commit 或本地修改，避免实际构建使用另一份依赖。

按完整上游的源码构建和调用指南，将生成包放入隔离 vendor 环境，然后运行 `activation/swi_glu/examples/test_aclnn_swi_glu.cpp` 和同目录 Host/Kernel UT。确认 OPP 和动态库解析实际指向本次构建结果；仅调用系统同名算子不能证明测到了这个固定源码。

数值参照先沿指定轴分成 A/B，在 FP32 中完成 `A/(1+exp(-A))*B`，最后转换输出类型。测试 FP16/BF16/FP32、M=1/16/256/8192、输出半宽 D=8/543/1024/4096，覆盖对齐、尾块、低精度和多轮 tile。

保存实际 `isDoubleBuffer`、baseRowLen/baseColLen、blockDim 和 tile 数。选择自然触发“双缓冲两 tile、单缓冲一 tile”条件的 shape，再在独立源码副本中改变 Host 选择条件并重建。不能只改一个下发标志而让 UB 大小与 Kernel 模板不匹配。

每次计时记录 SoC、CANN、驱动、源码/依赖 commit、shape、dtype、布局、warmup、计时口径和误差。排除编译、初始化与 H2D，分别保留 device duration 和完整调用时延。本交付未在 NPU 上编译或测量。
