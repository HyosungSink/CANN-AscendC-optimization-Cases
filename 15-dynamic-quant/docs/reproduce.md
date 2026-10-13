# 固定版本复现：dynamic_quant

## 现有验证边界

已完成源码下载、正式tag/commit核对、A2/A3注册核对、实际Kernel阅读和逐文件SHA256验证。当前macOS无NPU，**未编译/运行/计时NPU内核**。code/upstream是阅读快照，完整构建从固定仓库获取。

## 获取源码与依赖

在选定工作目录运行：

```bash
bash outputs/blog/15-dynamic-quant/code/reproduce.sh verify
bash outputs/blog/15-dynamic-quant/code/reproduce.sh fetch
```

默认源码在 `work/15-dynamic-quant/ops-nn`；ops-nn固定`0413e303e6cd1986b592c3e039f2579ac0f10b62`，opbase固定工程实际要求的`ec88904b611b04f9b23b529aee66c78bb7cf6140`。已有目录commit不一致时停止，避免覆盖另一份工程；可用`CASE_WORK_DIR`指定位置。

CANN Toolkit/ops/HDK须事先由用户按[官方9.0.0配套](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)准备。source实际安装的环境文件，检查`ASCEND_HOME_PATH`、`bisheng`和设备`npu-smi info`。A2用`ascend910b`，A3用`ascend910_93`。

## 离线依赖与构建

按[上游调用文档](https://gitcode.com/cann/ops-nn/blob/0413e303e6cd1986b592c3e039f2579ac0f10b62/docs/zh/invocation/quick_op_invocation.md)准备第三方源码归档，放在`work/15-dynamic-quant/third_party/pkg/`：json3.11.3的`include.zip`、`eigen-5.0.0.tar.gz`、`makeself-release-2.5.0-patch1.tar.gz`、`protobuf-25.1.tar.gz`、`abseil-cpp-20230802.1.tar.gz`。可设`THIRD_PARTY_DIR`使用现有缓存；缺失时wrapper停止，不让上游自动下载大包。

```bash
CANN_VERSION=9.0.0 SOC=ascend910b \
  bash outputs/blog/15-dynamic-quant/code/reproduce.sh build
```

它在完整工程执行等价于：

```bash
bash build.sh --pkg --soc=ascend910b --ops=dynamic_quant \
  --vendor_name=blog15 --cann_3rd_lib_path=/absolute/third_party -j8
```

A3改`SOC=ascend910_93`。成功产物在build_out，需保存真实构建日志；wrapper只构建，不装包、不sudo。源码与任意更新CANN不能保证混编，升级时重新使用配套tag。

## 调用指定的单个A2/A3样例

在测试环境按上游指引装入已构建的自定义包，source其vendor环境，记录安装路径和自定义库。自定义vendor的op_api目录应含include/lib；设置：

```bash
export CUSTOM_OPAPI_ROOT=/absolute/cann/opp/vendors/blog15_nn/op_api
CANN_VERSION=9.0.0 SOC=ascend910b \
  bash outputs/blog/15-dynamic-quant/code/reproduce.sh run-example
```

该模式只编译`quant/dynamic_quant/examples/test_aclnn_dynamic_quant.cpp`。安装布局不同需核对include/lib路径。以profiler确认加载的自定义kernel，API返回成功也可能只是调用内置版本。

**目标模板覆盖限制：特殊DS模板要求无group_index、INT8输出、行宽4096/5120/10240、行数4–384；默认示例未必满足这些条件。需要对照实际tilingKey，而非从算子名推断。**

## 从smoke test到数值/性能验证

按正文模板条件修改shape/dtype/转置/scale/layout，生成非退化随机输入，采用同量化契约或同FP32精度选项的参考实现。不能只把ND权重改shape当作NZ。记录误差、有效输出范围和实际tilingkey；任何改动在新分支记录diff与commit。

计时前预分配workspace、权重转换、H2D搬运和预热。用device事件或官方profiler统计重复执行的单算子device时间，列明中位数与离散程度，排除CPU初始化。针对正文的门槛/缓存选择做相邻shape实验，观察MTE/FIX/Vector/Cube与等待时间，再判断收益来源。

当前benchmark表留空：

| 设备SoC | Toolkit/ops/HDK | shape/dtype/layout | template/commit | device μs | 误差 | workspace |
| --- | --- | --- | --- | ---: | --- | ---: |
| 待测 | 待填 | 待填 | 待填 | 待测 | 待测 | 待测 |

本文尚无已完成的NPU结果，不能将源码机制推导标为实际加速成绩。
