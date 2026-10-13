# 复现 PTO TopK

目标：CANN 9.0.0、Atlas A2/910B 或 A3，Linux，PTO 正式 v9.0.0 commit `377526952e95c39f9149cb94d261eba9e394f0d4`。准备匹配驱动/固件、已安装的 bisheng、CMake、NumPy，并 source CANN 环境。正式版本证据见 `evidence/release-notes-9.0.0.md` 和原始 `version.cmake`。

## 源码与默认运行

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh checkout
SOC_VERSION=Ascend910B1 bash code/reproduce.sh run
```

`checkout` 明确下载固定源码，不安装包。可用 `SOURCE_DIR` 指定完整固定 checkout。`run` 在原始案例目录生成输入/golden，独立 `build-blog` 目录中 CMake 构建并执行 `topk`；不调用上游脚本的 `rm -rf build`。

默认规格 FP32、rows=4800、validCols=1024、physicalCols=1280、k=1000、48个block；每块100行，每组两行，两组交错。Host 默认 device=0。需要核对实际板型的核数和 SoC，不能把 A3 48-vector-core 时延直接当作 A2 时延。

## 正确性与设备计时

要求 stdout 显示 `All tests passed!`，并检查 value/index 两类输出；默认 Host 在失败时仍可能正常退出，因此不能只用进程 exit code 判断正确性。原始 golden 使用降序 value 和升序 index 的 lexsort，对重复值、FP16 ties 建议额外构造输入验证。

```bash
SOC_VERSION=Ascend910B1 bash code/reproduce.sh profile
```

profile 先构建、运行正确性，再用已安装 msprof 运行同一二进制。提取设备 TopK 事件，不使用 shell 编译总时间。Host 默认一次 launch，没有稳健的重复平均计时；若要报告平均值，需修改 Host 增加预热和重复样本，并保存修改 diff。

## 修改规格的边界

同时修改 `main.cpp`、`launchTopk` 模板实例、`scripts/gen_data.py` 的参数。检查 rows%blockDim=0、每block行数%4=0、topk不超过有效列、UB静态预算与打包格式。FP16公开表的 k=1008，不是把默认 FP32 k=1000 只换一个 dtype。

保留完整环境：NPU型号/核数、CANN/驱动/固件、源码commit、data dtype、k、有效列/物理跨度、blockDim、精度输出、profiler duration。公开表没有精确测量CANN和历史commit，不以324.106μs作为pass/fail阈值。

本机未执行 NPU 构建或测量，仅验证 source/evidence hash 与 shell 语法。
