# 复现 flash_attention_score_grad：完整工程与验收边界

## 当前状态

已从官方固定正式tag获取真实Kernel/Host/API文件，核对版本和设备注册并保存SHA256。macOS本机没有NPU，**未编译、运行、验证数值或测量NPU性能**。code/upstream是阅读快照，不是独立工程。

## 固定源码

从自己的工作目录运行：

```bash
bash outputs/blog/23-flash-attention-grad/code/reproduce.sh verify
bash outputs/blog/23-flash-attention-grad/code/reproduce.sh fetch
```

完整工程固定`ops-transformer afe72144f9f2ac8441929035795db88a111b30c5`，opbase固定`929902e577bb809a377a6ab17cb7b00a4dfddc1a`。默认写入当前目录`work/23-flash-attention-grad`；可设`CASE_WORK_DIR`。已有目录commit不匹配时停止，不重置另一份工程。build检查自身、缓存opbase及CMake优先的`../../ops-base`都没有dirty变更；实验代码需另外记录新commit、diff和pins。

## Linux/CANN/设备准备

由用户事先安装正式CANN 9.0.0 Toolkit、匹配ops与Ascend HDK，按[官方版本说明](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)配套。source实际安装环境，核对`ASCEND_HOME_PATH`、`bisheng`、g++、cmake、python3和`npu-smi info`。A2编译SoC为ascend910b，A3为ascend910_93。FlashAttention反向还要求受支持的训练系列产品。

## 准备离线第三方源码，构建单算子包

按官方quick_op_invocation准备json3.11.3的`include.zip`、`eigen-5.0.0.tar.gz`、`makeself-release-2.5.0-patch1.tar.gz`、`protobuf-25.1.tar.gz`、`abseil-cpp-20230802.1.tar.gz`，放在`work/23-flash-attention-grad/third_party/pkg/`。可设`THIRD_PARTY_DIR`；缺归档时wrapper停止，不让上游自动下载大包。

```bash
CANN_VERSION=9.0.0 SOC=ascend910b \
  bash outputs/blog/23-flash-attention-grad/code/reproduce.sh build
```

上游实际命令为：

```bash
bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score_grad \
  --vendor_name=blog23 --cann_3rd_lib_path=/absolute/third_party -j8
```

A3改为SOC=ascend910_93。wrapper只构建build_out产物，不自动安装包、下载CANN或sudo。保留实际构建日志；只有真实成功才更新“编译通过”。

## 指定样例与模板覆盖

按上游文档在测试环境装入生成的自定义包，source其vendor环境，确定op_api目录含include/lib，并记录实际路径：

```bash
export CUSTOM_OPAPI_ROOT=/absolute/cann/opp/vendors/blog23_transformer/op_api
CANN_VERSION=9.0.0 SOC=ascend910b \
  bash outputs/blog/23-flash-attention-grad/code/reproduce.sh run-example
```

这只编译`attention/flash_attention_score_grad/examples/test_aclnn_flash_attention_score_grad_v2.cpp`，不遍历混合950样例。安装布局不同需要核对include/lib路径。profiler应确认自定义包路径与内核，API成功不证明选中了正文模板。

支持A2/A3训练系列，产品表不支持相应推理系列。官方普通样例用常量Softmax统计，只能作为API调用样例；梯度验证必须使用真实前向结果/Softmax max/sum及同一Dropout mask。SAMEAB一般路径不支持FP32，默认FP32示例不覆盖目标模板。

## 数值与性能闭环

按正文shape条件生成非退化随机输入，以匹配数学契约的参考实现比较有效输出。可变形卷积要同时检查四邻点插值、mask/padding、deformable与conv两种group、bias与API布局变换；FlashAttention反向要先运行匹配前向，取得真实Y、max/sum，Dropout复用同一个mask，检查dQ/dK/dV及需要的确定性结果。

预先完成分配、转置、权重转换、H2D、预热；用device事件或官方profiler测重复执行。核心Kernel与整条aclnn链分列计时，避免Cast/Transpose/Add等API操作混入机制分析。记录模板名、tilingkey、实际核数、workspace、MTE/FIX/Vector/Cube时间、误差和device时延分布。

当前单shape实测表为空：

| 设备/SoC | Toolkit/ops/HDK | shape/dtype/layout | template/commit | device μs | 误差 | workspace |
| --- | --- | --- | --- | ---: | --- | ---: |
| 待测 | 待填 | 待填 | 待填 | 待测 | 待测 | 待测 |

FlashAttention版本级1%+/10%+若要重现，须分别固定8.5.0与9.0.0的匹配工程/软件栈和完整shape集合；不要把新版本内单个开关实验叫作复现官方泛化平均。
