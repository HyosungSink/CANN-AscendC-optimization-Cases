# CANN 9.0.0版本说明

## 版本下载地址

<https://www.hiascend.com/cann/download>

## 版本配套

**1、CANN与Ascend HDK版本配套关系**

|CANN版本  |  配套Ascend HDK版本| 
|--|--|
| [CANN 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) | [Ascend HDK 26.0.RC1 ](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=All&driver=Ascend+HDK+26.0.RC1)<br> [Ascend HDK 25.5.2 ](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=All&driver=Ascend+HDK+25.5.2)  <br> [Ascend HDK 25.5.1](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=All&driver=Ascend+HDK+25.5.1) |
| [CANN 8.5.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2) | [Ascend HDK 26.0.RC1 ](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=All&driver=Ascend+HDK+26.0.RC1) <br> [Ascend HDK 25.5.2 ](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=All&driver=Ascend+HDK+25.5.2) <br> [Ascend HDK 25.5.1 ](https://www.hiascend.com/hardware/firmware-drivers/community?product=4&model=32&cann=All&driver=Ascend+HDK+25.5.1)|


**2、 CANN组合包版本配套关系**

|ops版本  |配套toolkit版本  |  
|--|--|
| [ascend-cann-ops 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) | [ascend-cann-toolkit 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0)<br>[ascend-cann-toolkit 8.5.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2) |
| [ascend-cann-ops 8.5.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2)  |  [ascend-cann-toolkit 9.0.0](https://www.hiascend.com/developer/download/community/result?module=cann&cann=9.0.0) <br> [ascend-cann-toolkit 8.5.2](https://www.hiascend.com/developer/download/community/result?module=cann&cann=8.5.2) |

**3、 CANN独立升级子包版本配套关系**

| CANN子包版本                                                 | 版本源码标签                                                 | 配套CANN版本      |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------- |
| [cann-ops-math 9.0.0](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0/) | [v9.0.0](https://gitcode.com/cann/ops-math/tags/v9.0.0)      | CANN 9.0.0<br>CANN 8.5.2 |
| [cann-ops-nn 9.0.0](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0/) | [v9.0.0](https://gitcode.com/cann/ops-nn/tags/v9.0.0)        | CANN 9.0.0<br>CANN 8.5.2 |
| [cann-ops-cv 9.0.0](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0/) | [v9.0.0](https://gitcode.com/cann/ops-cv/tags/v9.0.0)        | CANN 9.0.0<br>CANN 8.5.2 |
| [cann-ops-transformer 9.0.0](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0/) | [v9.0.0](https://gitcode.com/cann/ops-transformer/tags/v9.0.0) | CANN 9.0.0<br>CANN 8.5.2 |
| [cann-hccl 9.0.0](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0/) | [v9.0.0](https://gitcode.com/cann/hccl/tags/v9.0.0)          | CANN 9.0.0<br>CANN 8.5.2 |
| [cann-hixl 9.0.0](https://ascend.devcloud.huaweicloud.com/cann/run/software/9.0.0/) | [v9.0.0](https://gitcode.com/cann/hixl/tags/v9.0.0)          | CANN 9.0.0<br>CANN 8.5.2 | 

**4、CANN开源子包版本配套关系**

| CANN子包版本                       | 版本源码标签                                                 | 配套CANN版本      |
| ---------------------------------- | ------------------------------------------------------------ | ----------------- |
| cann-opbase 9.0.0           | [v9.0.0](https://gitcode.com/cann/opbase/tags/v9.0.0)        | CANN 9.0.0   |
| cann-oam-tools 9.0.0        | [v9.0.0](https://gitcode.com/cann/oam-tools/tags/v9.0.0)     | CANN 9.0.0   |
| cann-asc-tools 9.0.0        | [v9.0.0](https://gitcode.com/cann/asc-tools/tags/v9.0.0)     | CANN 9.0.0   |
| cann-asc-devkit 9.0.0       | [v9.0.0](https://gitcode.com/cann/asc-devkit/tags/v9.0.0)    | CANN 9.0.0   |
| cann-pto-isa 9.0.0          | [v9.0.0](https://gitcode.com/cann/pto-isa/tags/v9.0.0)       | CANN 9.0.0   |
| cann-ge-compiler 9.0.0      | [v9.0.0](https://gitcode.com/cann/ge/tags/v9.0.0)            | CANN 9.0.0   |
| cann-ge-executor 9.0.0      | [v9.0.0](https://gitcode.com/cann/ge/tags/v9.0.0)            | CANN 9.0.0   |
| cann-graph-autofusion 9.0.0 | [v9.0.0](https://gitcode.com/cann/graph-autofusion/tags/v9.0.0) | CANN 9.0.0   |
| cann-metadef 9.0.0          | [v9.0.0](https://gitcode.com/cann/metadef/tags/v9.0.0)       | CANN 9.0.0   |
| cann-dflow-executor 9.0.0   | [v9.0.0](https://gitcode.com/cann/ge/tags/v9.0.0)            | CANN 9.0.0   |
| cann-hcomm 9.0.0            | [v9.0.0](https://gitcode.com/cann/hcomm/tags/v9.0.0)         | CANN 9.0.0   |
| cann-npu-runtime 9.0.0      | [v9.0.0](https://gitcode.com/cann/runtime/tags/v9.0.0)     | CANN 9.0.0   |

子包独立升级的具体操作请参考[子包独立升级](#子包独立升级)。

## 关键特性
### 支持Ascend 950PR款型
CANN新增适配Ascend 950PR（Atlas 350加速卡）开源并新增相关重点特性：支持FP8/MxFP8/MxFP4、AscendC支持SIMD+SIMT混合编程、集合通信支持CCU通信加速。

### 安装易用性提升：新增apt和pip安装方式，提供一站式下载安装能力

新增**apt**和**pip**安装方式。当前已支持conda、yum、apt、pip四种主流安装方式。同时，新增**CANN一站式下载安装能力**，软件获取与部署时长从2h缩短至45min。[特性链接](https://www.hiascend.com/cann/download)

### DeepSeek系列加速：DSA系列算子性能优化并支持确定性计算

在Atlas A2 系列产品和Atlas A3 系列产品上，DSA系列算子进一步提升流水并行度，其中SFAG算子性能提升至2x~6x+，模型提升至1.5x+。新增支持确定性计算能力，可满足对结果一致性要求较高的场景需求。[特性链接](https://gitcode.com/cann/ops-transformer/blob/9.0.0/attention/sparse_flash_attention_grad/docs/aclnnSparseFlashAttentionGrad.md)

### 多模态场景加速：新增块稀疏Attention算子支持

在Atlas A2 系列产品和Atlas A3 系列产品上，新增支持块稀疏Attention算子（Block-wise Sparse Attention,BSA），相比通用Flah Attention，90%稀疏率情况下，BSA计算量降低90%，执行速度提升至5x。wan2.2和hunyuanvideo1.5在70%稀疏率时，模型分别提升至1.7x和1.5x。[特性链接](https://gitcode.com/cann/ops-transformer/blob/9.0.0/attention/block_sparse_attention/README.md)

### 小消息高频通信场景加速：HCCL支持批量通信合并机制

在Atlas A2 系列产品和Atlas A3 系列产品上，HCCL新增支持批量通信合并机制（HcclGroupStart/HcclGroupEnd接口），支持将多个通信操作合并后统一提交与执行，提升小消息、高频通信场景下的执行效率。([HcclGroupStart](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclcpp_07_0063.html),[HcclGroupEnd](https://www.hiascend.com/document/detail/zh/canncommercial/900/API/hcclug/hcclcpp_07_0064.html))

## 新增特性

### 公共模块

- CANN提升安装部署的易用性：
  - CANN新增apt-get、pip在线安装方式。
  - CANN增加合一包：合一包由昇腾NPU驱动和Toolkit组成，支持一键式安装和通过--whitelist参数指定安装。
- CANN的ops算子包下新增Ascend-cann-950-ops包，支持Ascend 950PR算子功能。
- CANN适配不同的昇腾硬件产品，新增OS兼容性支持。
  - 新增支持Atlas 350 加速卡，并适配veLinux 2.0、Alinux 3。
  - Atlas 200T A2 Box16适配Ubuntu 22.04.5。
  - Atlas 800T A2、Atlas 800I A2、Atlas 900 A2 PoDc 适配UOS V25（6.6内核）。
  - Atlas 800I A2适配AntOS（6.6内核）。
  - Atlas 800T A3、Atlas 900 A3 SuperPoD适配CTyunOS4-25.07、UOS V25（6.6内核）。
- CANN支持numpy 2.x 版本，支持CANN在高版本numpy下运行。

### 算子库

#### ops-nn库

- lstm系列算子开源:
  - [ThnnFusedLstmCellGrad算子]（[!793](https://gitcode.com/cann/ops-nn/pull/793)）。
  - [SingleLayerLstmGrad算子]（[!796](https://gitcode.com/cann/ops-nn/pull/796)）。
  - [ThnnFusedLstmCell算子]（[!1999](https://gitcode.com/cann/ops-nn/pull/1999)）。
- 低bit类算子和融合算子支持更多数据类型：fp8/mxfp8/hifp8/mxfp4等，并支持pertensor/perchannel/pertoken/pergroup/perblock等不同量化和组合方式：
  - [全量化融合算子:quant_batch_matmul_v4](https://gitcode.com/cann/ops-nn/blob/9.0.0-beta.1/matmul/quant_batch_matmul_v4/README.md)。
  - [伪量化融合算子:weight_quant_batch_matmul_v2](https://gitcode.com/cann/ops-nn/blob/9.0.0-beta.1/matmul/weight_quant_batch_matmul_v2/README.md)。
  - [mx动态量化算子:dynamic_mx_quant](https://gitcode.com/cann/ops-nn/blob/9.0.0-beta.1/quant/dynamic_mx_quant/README.md)。
  - [非mx动态量化算子:dynamic_quant_v2](https://gitcode.com/cann/ops-nn/blob/9.0.0-beta.1/quant/dynamic_quant_v2/README.md)。
  - [mx动态量化算子:grouped_dynamic_mx_quant](https://gitcode.com/cann/ops-nn/blob/9.0.0-beta.1/quant/grouped_dynamic_mx_quant/README.md)。
  - [非mx动态量化算子:grouped_dynamic_block_quant](https://gitcode.com/cann/ops-nn/blob/9.0.0-beta.1/quant/grouped_dynamic_block_quant/README.md)。
- 开源算子支持Ascend 950PR:
  - 算子工程适配（[!450](https://gitcode.com/cann/ops-nn/pull/450)）。
  - LogSigmoid（[!1837](https://gitcode.com/cann/ops-nn/pull/1837)）。
  - Norm类（[!995](https://gitcode.com/cann/ops-nn/pull/995)）。
  - SoftMax类（[!1010](https://gitcode.com/cann/ops-nn/pull/1010)）。
  - AvgPoolV2Grad（[!1183](https://gitcode.com/cann/ops-nn/pull/1183)）。
  - RmsNormQuant（[!1057](https://gitcode.com/cann/ops-nn/pull/1057)）。
- Ascend 950PR支持Parallel Welford和超长轴二分累加算法，提高Norm类算子的数值稳定性与计算精度，该特性主要涉及算子：
  - [LayerNormV4](https://gitcode.com/cann/ops-nn/tree/master/norm/layer_norm_v4)
  - [BatchNormV3](https://gitcode.com/cann/ops-nn/tree/master/norm/batch_norm_v3)
- Ascend 950PR新增MM融合算子： 
  - [QuantBatchMatmulInplaceAdd]（[!1130](https://gitcode.com/cann/ops-nn/pull/1130)）。
  - [TransposeQuantBatchMatmul]（[!1136](https://gitcode.com/cann/ops-nn/pull/1136)）。
  - [DualLevelQuantBatchMatmul]（[!1141](https://gitcode.com/cann/ops-nn/pull/1141)）。
- `QuantBatchMatmul`算子基于Ascend 950PR支持mxfp8 weightNz特性，提升网络推理性能（[!1144](https://gitcode.com/cann/ops-nn/pull/1144)）。
- 支持SIMD/SIMT新同构编程算子实现:
  - [MapIndex]（[#660](https://gitcode.com/cann/ops-nn/issues/660)）。
  - [ScatterSub]（[#710](https://gitcode.com/cann/ops-nn/issues/710)）。
  - [SegmentSum]（[#668](https://gitcode.com/cann/ops-nn/issues/668)）。
  - [UnsortedSegmentMin]（[#656](https://gitcode.com/cann/ops-nn/issues/656)）。
  - [opKTopPSampleV2]（[#658](https://gitcode.com/cann/ops-nn/issues/658)）。
- 新增<<<>>>的算子开发样例（[!620](https://gitcode.com/cann/ops-nn/pull/620)）。
- 支持编译生成静态库，发布的nn独立组件包携带静态库（[!391](https://gitcode.com/cann/ops-nn/pull/391)）。
- 新增支持按模板编译特定算子kernel能力（[#1097](https://gitcode.com/cann/ops-nn/issues/1097)）。
- 提供matmul、卷积、norm和hash类算子编程的优秀实践： 
  - [MatMul算子VCV性能优化实践与效果分析](https://gitcode.com/cann/ops-nn/wiki/MatMul%E7%AE%97%E5%AD%90VCV%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%E4%B8%8E%E6%95%88%E6%9E%9C%E5%88%86%E6%9E%90.md)。
  - [MatMul算子性能优化实践与效果分析](https://gitcode.com/cann/ops-nn/wiki/MatMul%E7%AE%97%E5%AD%90%E6%80%A7%E8%83%BD%E4%BC%98%E5%8C%96%E5%AE%9E%E8%B7%B5%E4%B8%8E%E6%95%88%E6%9E%9C%E5%88%86%E6%9E%90.md)。
- 新增支持LSTM/LSTMCell通过aclnn接口调用，能显著缩短使用LSTM算子的语音处理等模型通过torch api调用时，host侧的编译耗时。
- aclnnScatter/aclnnScatterValue/aclnnScatterAdd等索引类接口实现dim!=-1场景的内存优化。对于使用scatter或scatter_add接口的模型训练或推理场景，在dim!=-1时有明显内存收益。

#### ops-transformer库

- transformer相关算子在Atlas A3系列产品上能力完善：
  - [MC2 dispatch和combine算子支撑Aiv直驱Roce能力:moe_distribute_dispatch](https://gitcode.com/cann/ops-transformer/blob/9.0.0-beta.1/mc2/moe_distribute_dispatch_v2/README.md)。
  - [Rope 支持算子泛化mrope_section](https://gitcode.com/cann/ops-transformer/blob/9.0.0-beta.2/posembedding/rope_with_sin_cos_cache/README.md)。
  - [BlockSparseAttention算子能力增强](https://gitcode.com/cann/ops-transformer/blob/9.0.0-beta.2/attention/block_sparse_attention/README.md)。
  - [GroupedMatmulSwigluQuantV2算子能力增强支持A4W4动态分块优化](https://gitcode.com/cann/ops-transformer/blob/9.0.0-beta.2/gmm/grouped_matmul_swiglu_quant_v2/README.md)。
  - [GroupedMatmul算子能力增强A4W4支持NZ转置动态分块优化](https://gitcode.com/cann/ops-transformer/blob/9.0.0-beta.2/gmm/grouped_matmul/README.md)。
- DSA系列算子支持确定性计算。SparseFlashAttentionGrad、LightningIndexerGrad、DenseLightningIndexerGradKLLoss等算子支持确定性计算。对于DeepSeekV3.2模型训练，开启确定性计算，Attention部分可保证相同输入多次执行结果一致。
- 新增MatmulAllToAll、AllToAllMatmul通算融合算子。对于采用Ulysses并行的M级超长序列训练模型，使用该算子可提升模型性能。
- GmmSwigluQuantV2 支持i4进i8出、A4W4动态分块优化。Gmm A4W4 支持NZ转置 动态分块优化
- Moe permute RoutingMap系列支持bf16token时，传入的probs为fp32。MoeInitRoutingQuantV2算子支持INT4和SmoothQuant。
- M-RoPE融合算子泛化mrope_section支持[16, 16, 16, 16]、[24, 20, 20]，mrope算子 prefill 性能优化。
- aclnnScatterPaKvCache支持FIA算子NZ场景，模型侧使用FIA算子能获取NZ使能的性能收益。
- FAG SAMEAB模板和BN2模板基础API优化重写，相对CANN 8.5.0 版本，泛化测试平均优化1%+和10%+。
- FAG部分场景（layout=BSND/BNSD/SBH，sparsemode=0/2/3，dtype=bfloat16，S1=S2），对其确定性性能进行了优化，这些场景可提升模型确定性性能。
- 新增Floyd attention正反向算子（Fuse Floyd Attention/Fuse Floyd Attention Grad），相比Floyd结构的小算子拼接，通过流水混排10个matmul，融合性能优化30% +，内存占用降低 70%+。
- RingAttentionUpdate算子支持Ascend 950PR。

#### ops-cv库

- image类算子支持Ascend 950PR，覆盖 `ResizeNearestNeighborV2`、`ResizeNearestNeighborV2Grad`、`UpsampleNearest`、`UpsampleNearest3d` 等（[!284](https://gitcode.com/cann/ops-cv/pull/284) [#97](https://gitcode.com/cann/ops-cv/issues/97) [#93](https://gitcode.com/cann/ops-cv/issues/93)）。
- CI对Ascend 950PR的编译与 UT 识别能力增强（ [!281](https://gitcode.com/cann/ops-cv/pull/281)、[!368](https://gitcode.com/cann/ops-cv/pull/368)、[#105](https://gitcode.com/cann/ops-cv/issues/105)）。
- 算子的example支持在仿真（simulator）上执行，可在不依赖真实硬件的场景下进行样例编译与执行验证，便于Ascend 950PR适配前置联调与回归（[!567](https://gitcode.com/cann/ops-cv/pull/567) [!397](https://gitcode.com/cann/ops-cv/pull/397)）。
- 新增RoiPoolingWithArgMax、RoiPoolingGradWithArgMax目标检测类算子，支持Ascend 950PR，支撑faster-rcnn模型功能打通。
- 新增支持CIou通过aclnn接口调用，支持Ascend 950PR上mmcv功能打通。
- 新增Col2im目标检测算子，支持Ascend 950PR，支撑智驾模型BEVfusion模型训练。
- 新增Rasterizer算子实现光栅化计算，根据三维空间点和面计算每个像素点的最小深度及其对应的面片索引，对于多模态3D生成场景使用该算子可提升模型性能。

#### ops-math库

- 数学运算、张量变换及随机数生成支持Ascend 950PR，覆盖`Erfc`、`Sinh`、`Asin`、`Atanh`、`BitwiseXor`、`Asinh`、`Cosh`、`Scale`、`Tan`、`Acos`、`Acosh`等（[#599](https://gitcode.com/cann/ops-math/issues/599)）。
- aclnnMul/aclnnMuls/aclnnAdd/aclnnAdds/aclnnSum等API支持非连续输入，使相关接口性能得到提升、内存占用有优化（[!874](https://gitcode.com/cann/ops-math/pull/874)）。
- Sort算子在Ascend 950PR上相较于Atlas A3，新增UINT16/UINT32/UINT64类型。Atlas A3支持的数据类型在Ascend 950PR性能平均提升1.5+倍（[#557](https://gitcode.com/cann/ops-math/pull/557) [#632](https://gitcode.com/cann/ops-math/pull/632)）。
- 离散类张量变换类算子性能优化，覆盖`Pad`、`Transpose`、`AsStrided` （[#569](https://gitcode.com/cann/ops-math/issues/569) [#539](https://gitcode.com/cann/ops-math/pull/539) [#495](https://gitcode.com/cann/ops-math/pull/495)）。
- 算子的example支持在仿真（simulator）上执行，可在不依赖真实硬件的场景下进行样例编译与执行验证，便于Ascend 950PR适配前置联调与回归（[!563](https://gitcode.com/cann/ops-math/pull/563) [!1215](https://gitcode.com/cann/ops-math/pull/1215)）。
- 算子工程支持kernel并行编译（[!779](https://gitcode.com/cann/ops-math/pull/779)）。
- 算子工程日志优化（[#192](https://gitcode.com/cann/ops-math/issues/192)）。
- 算子CI支持算子增量UT与冒烟（[!774](https://gitcode.com/cann/ops-math/pull/774) [!817](https://gitcode.com/cann/ops-math/pull/817)）。
- 算子构建脚本支持导出预编译文件，降低算子问题定位难度（[#536](https://gitcode.com/cann/ops-math/issues/536)）。
- 支持Docker部署（[!547](https://gitcode.com/cann/ops-math/pull/547)）。
- 新增ChunkCat融合算子，在FSDP2框架下训练时需要该算子做权重合并，相较于小算子拼接，内存降低30%，减少模型峰值显存，同时能够提升模型性能。
- aclnnDiv/aclnnDivs接口支持混合数据类型输入，优化算子内存占用。
- 新增logspace算子aclnn接口调用，提升接口易用性。
- aclnnReplicationPad3d补充支持bfloat16数据类型，提升多模态生成场景接口易用性。
- aclnnReplicationPad2dBackward接口内存优化，减少峰值内存使用，提升算子易用性。

#### opbase库

- 基础框架库支持Ascend 950PR（[#65](https://gitcode.com/cann/opbase/issues/65)）。

### 通信库

#### 集合通信

- 通信算子支持Ascend 950PR，覆盖 `Allgather`、`AllgatherV`、`Allreduce`、`AlltoAll`、`AlltoAllV`、`Broadcast`、`Reduce`、`ReduceScatter`、`ReduceScatterV`、`Scatter`、`SendRecv` 等([\#106](https://gitcode.com/cann/hccl/pull/106))。
- 在Atlas A3 系列产品的超节点内AICPU场景下 AlltoAll算子支持使用对称内存功能([\#575](https://gitcode.com/cann/hcomm/pull/575))。
- 在Atlas A3 系列产品的AICPU场景下，ReduceScatter&AllReduce算子支持Batch一致性([\#483](https://gitcode.com/cann/hcomm/pull/483))。
- 在Atlas A3 系列产品的超节点内 ReduceScatter&AllReduce&AllGather&AlltoAll算子支持多机间的superkernel([\#596](https://gitcode.com/cann/hcomm/pull/596))。
- 支持离线编译，提升构建易用性([\#126](https://gitcode.com/cann/hccl/pull/126))。
- 在Atlas A2 系列产品和Atlas A3 系列产品上，HCCL支持批量通信合并机制（HcclGroupStart/HcclGroupEnd接口），支持将多个通信操作合并后统一提交与执行。该机制可提升小消息、高频通信场景下的执行效率，也可以更好地支持多个点对点通信的统一调度。
- 在Atlas A2 系列产品上，单边通信新增支持AICPU加速模式。在PD分离部署场景下，相比于当前host cpu调度模式，推理pullkvcache数据量小于128K时，AICPU加速模式性能提升50%+。
- 优化Atlas A2 系列产品上跨机确定性算法性能，性能提升2x，相比于未开确定性计算场景，劣化<30%。在确定性场景下，有明显收益。
- 在Atlas A3 系列产品上支持按通信域粒度配置HCCL展开方式（AICPU/AIV等），提升配置灵活性。
- 在Atlas A3 系列产品上，allreduce/allgather/reducescatter/alltoall算子支持对称内存特性，提升算子执行性能。
- 在Atlas A3 系列产品上，allreduce/reducescatter算子在AICPU展开模式时支持batch一致性。
- 支持同一物理机内多容器间（不共卡）的部署方式，通信域初始化时使用随机选端口，多线程并发通信时context隔离等。
- 提升维测能力：ErrorMsg输出内容优化，AICPU展开新增时间打点等。

#### 单边通信

- 单边通信： Atlas A3 训练系列产品/Atlas A3 推理系列产品 支持自动建链模式，无需显式调用connect接口([\#106](https://gitcode.com/cann/hixl/issues/106))。
- 单边通信： Atlas A3 训练系列产品/Atlas A3 推理系列产品 支持Fabric Memory全局统一编址模式，支持使用HCCS进行D2rH的KV Cache直传能力，提升传输效率([\#33](https://gitcode.com/cann/hixl/issues/33))。

### 领域加速库

#### Ascend Transformer Boost加速库

- 核心算子（Activation、MultiLatentAttention、LayerNorm等多个算子）等支持Ascend 950PR。
- 整图下沉模式下， 支持对plugin算子更新参数。

#### Ascend Signal Processing Boost加速库

- SiP信号库新增torch extension模块，通过Torch Library绑定C++接口，支持通过python调用FFT、BLAS、SOLVER等40+算子能力。
- 优化内存分配管理机制，通过资源池实现高效的内存动态管理，在FFT场景下提高handle的复用，减少资源浪费和重复申请释放。

### 图引擎

- ES构图提供多种场景的sample ([!72](https://gitcode.com/cann/ge/pull/72) 、[!123](https://gitcode.com/cann/ge/pull/123)) 。
- ES构图 Readable Dump支持子图的友好展示。
- 自定义pass改图能力增强，新增原图优化之后改图的能力。
- 图模式多流场景nopadding连续内存支持内存复用。
- ES构图支持消费历史原型生成合法的重载cxx接口（[\#643](https://gitcode.com/cann/ge/pull/643)）。
- 支持算子级控核场景代码优化（[\#450](https://gitcode.com/cann/ge/pull/450)）。
- 图融合删除模型输出节点时，框架支持识别更新模型输出（[\#434](https://gitcode.com/cann/ge/pull/434)）。
- 公共子表达式消除优化（[\#622](https://gitcode.com/cann/ge/pull/622)）。
- 支持通过dump开关使能L0和L1 exception dump（[\#398](https://gitcode.com/cann/ge/pull/398)）。
- error msg优化整改（[\#685](https://gitcode.com/cann/ge/pull/685)）。
- 提供端到端Sample：动态分档样例（[\#813](https://gitcode.com/cann/ge/pull/813)）（[\#685](https://gitcode.com/cann/ge/pull/685)），自定义算子入图样例（[\#867](https://gitcode.com/cann/ge/pull/867)），基于graph接口的Matmul+Add融合为GEMM自定义pass样例（[\#1106](https://gitcode.com/cann/ge/pull/1106)）。
- 编译工程优化（[\#646](https://gitcode.com/cann/ge/pull/646)）（[\#890](https://gitcode.com/cann/ge/pull/890)）。
- 支持确定性和强一致性配置：ge.deterministicLevel。
- 支持TensorMove消除。
- 在线场景支持用户使用graph级别的option指定模型输出datatype：ge.outputDatatype。
- 去除执行路径上的单例和锁，提升调度性能。

### 算子编程

- Ascend 950PR支持SIMD编程模式，提供[200+ API 接口](https://gitcode.com/cann/asc-devkit/tree/9.0.0/impl/basic_api/dav_c310)跨代兼容能力，可实现Atlas A2系列产品和Atlas A3系列产品算子平滑迁移。
- Ascend 950PR新增基于Reg的编程方式，提供Reg数据搬运、基础算术、规约计算、同步控制等[90+ Reg编程接口](https://gitcode.com/cann/asc-devkit/tree/9.0.0/impl/basic_api/reg_compute/dav_c310)。
- Atlas A2系列产品、Atlas A3系列产品、Ascend 950PR支持[语言扩展层纯 C 接口](https://gitcode.com/cann/asc-devkit/tree/9.0.0/include/c_api)，支持数组式内存分配与指针型计算接口，提供原生纯 C 编程体验。
- Ascend 950PR支持SIMD与SIMT混合编程，提供约700个[SIMT API接口](https://gitcode.com/cann/asc-devkit/tree/9.0.0/include/simt_api)，包含warp、atomic、基本数学计算、类型转换等基础接口。
- Ascend 950PR支持通信高阶API的CCU通信接口，提供基于CCU的[Allreduce，Allgather，Reducescatter，AlltoAll等主流通信原语](https://gitcode.com/cann/asc-devkit/tree/9.0.0/impl/adv_api/detail/hccl/impl/platform_v310)；Matmul高阶API新增支持[MXFP4/8低比特数据类型的矩阵运算](https://gitcode.com/cann/asc-devkit/blob/9.0.0/impl/adv_api/detail/matmul/mx_matmul_impl.h)，实现内存占用减半、算力吞吐倍增。
- Ascend 950PR新增及兼容支持样例共计约260个，包含SIMT样例、SIMD样例（框架类、基础API、高阶API、最佳实践等），并按照编程模型和样例类别对[样例目录结构进行调整](https://gitcode.com/cann/asc-devkit/pull/1223)，提升样例目录结构的易读性。
- 完成[Gitee样例仓](https://gitee.com/ascend/samples/tree/master/operator/ascendc)AscendC样例及[AscendC高阶API仓](https://gitee.com/ascend/ascendc-api-adv/tree/master/examples)样例到[asc-devkit仓](https://gitcode.com/cann/asc-devkit/tree/9.0.0/examples)的迁移，并统一使用<<<>>>调用方式；
- 融合编译与<<<>>>调用方式支持[CPU模式](https://gitcode.com/cann/asc-tools/pull/138)以及[SIM仿真模式](https://gitcode.com/cann/asc-devkit/blob/9.0.0/cmake/asc/asc_modules/CMakeASCInformation.cmake)。
- 联合毕昇编译器，优化融合编译性能。

### 虚拟指令集

- 支持[昇腾Ascend 950PR指令集](https://gitcode.com/cann/pto-isa/tree/master/include/pto/npu/a5)（含Element-Wise、TileScalar、固定管线、访存操作、复杂操作、卷积指令、量化指令等）及对应指令的[CPU-SIM实现](https://gitcode.com/cann/pto-isa/tree/master/include/pto/cpu)
- 新增SDMA([#493](https://gitcode.com/cann/pto-isa/pull/493))、URMA([#715](https://gitcode.com/cann/pto-isa/pull/715))异步通信指令，多卡P2P/P2MP/信号量同步操作指令([#17]((https://gitcode.com/cann/pto-isa/pull/17)))
- 新增[基础指令的CostModel](https://gitcode.com/cann/pto-isa/tree/master/include/pto/costmodel)性能仿真
- 新增调试指令TPRINT，支持Vec、Mat、Acc存储中Tile数据及GM中Tensor数据打印([#31](https://gitcode.com/cann/pto-isa/pull/31), [#638](https://gitcode.com/cann/pto-isa/pull/638), [#725](https://gitcode.com/cann/pto-isa/pull/725))
- 新增性能调优指令TPUSH/TPOP([#364](https://gitcode.com/cann/pto-isa/pull/364), [#431](https://gitcode.com/cann/pto-isa/pull/431), [#569](https://gitcode.com/cann/pto-isa/pull/569))及TPREFETCH([#73](https://gitcode.com/cann/pto-isa/pull/73), [#116](https://gitcode.com/cann/pto-isa/pull/116))

### 运行时

- 运行时Runtime支持Ascend 950PR。
- 易用性增强：
  - 支持AclGraph场景stream规格扩充至64k，解决大模型资源不足等问题（[\#461](https://gitcode.com/cann/runtime/pull/461)）。
  - 支持Aclgraph场景Event规格扩充，扩充后的Event规格仅取决于Device内存（[\#482](https://gitcode.com/cann/runtime/pull/482)）。
  - 提供包版本号查询接口，根据包名查询返回数值版本号和字符串版本号，接口如下： 
    - [aclError aclsysGetVersionStr(char *pkgName, char * versionStr)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclsysGetVersionStr.md) 
    - [aclError aclsysGetVersionNum(char *pkgName，int32_t * versionNum)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclsysGetVersionNum.md) 

  - 支持查询指定流（Stream）的优先级，接口如下： 
    - [aclError aclrtStreamGetPriority(aclrtStream stream, uint32_t *priority)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclrtStreamGetPriority.md) 

  - 支持查询创建Stream时设置的flag标志，接口如下： 
    - [aclError aclrtStreamGetFlags(aclrtStream stream, uint32_t *flags)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclrtStreamGetFlags.md)   

  - 支持获取Device的唯一标识UUID（Universally Unique Identifier），接口如下： 
    - [aclError aclrtDeviceGetUuid (int32_t deviceId, aclrtUuid *uuid)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclrtDeviceGetUuid.md) 
 
  - 支持获取待查询地址所属内存块的起始地址以及内存块大小，接口如下： 
    - [aclError aclrtMemGetAddressRange(void *ptr, void **pbase, size_t *psize)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclrtMemGetAddressRange.md) 
 
  - 支持设置和查询强一致性计算的参数，相关接口如下：
    - [aclError aclrtSetSysParamOpt(aclSysParamOpt opt, int64_t value)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclrtSetSysParamOpt.md) 
    - [aclError aclrtGetSysParamOpt(aclSysParamOpt opt, int64_t *value)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclrtGetSysParamOpt.md) 
    - [aclError aclrtCtxSetSysParamOpt(aclSysParamOpt opt, int64_t value)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclrtCtxSetSysParamOpt.md) 
    - [aclError aclrtCtxGetSysParamOpt(aclSysParamOpt opt, int64_t *value)](https://gitcode.com/cann/runtime/blob/9.0.0/docs/api_docs/aclrtCtxGetSysParamOpt.md)
- 发布Runtime编程指南（[\#1030](https://gitcode.com/cann/runtime/pull/1030)）。
- 发布Runtime Ascend 950PR配套资料（[\#1255](https://gitcode.com/cann/runtime/pull/1255)）。

### 开发与维测工具

#### 性能调优工具

- msprof支持aicore-metrics选项采集自定义PMU指标能力（[\#136](https://gitcode.com/cann/oam-tools/pull/136)）.
- HCCL性能测试工具支持FP64数据类型（[\#122](https://gitcode.com/cann/oam-tools/pull/122)）.

#### AMCT模型压缩工具

- 支持HIF8/FP8/FP4/MXFP8/MXFP4量化数据类型，支持HIF8 OFMR量化算法（[\#20](https://gitcode.com/cann/amct/pull/20)）
- 支持HIF8分位量化算法（[\#54](https://gitcode.com/cann/amct/pull/54)）

## 删除和废弃特性

### 算子库

**transformer库以下接口在CANN 9.0.0中被标记为废弃，将在2027年3月30日之后的版本删除**
- aclnnGroupedMatMulAllReduce接口废弃，替换为：aclnnMatmulAllReduce。
- aclnnGroupedMatmul/aclnnGroupedMatmulV2/aclnnGroupedMatmulV3/aclnnGroupedMatmulV4接口废弃，替换为：aclnnGroupedMatmulV5。
- aclnnFusedInferAttentionScore/aclnnFusedInferAttentionScoreV2/aclnnFusedInferAttentionScoreV3接口废弃，替换为：aclnnFusedInferAttentionScoreV4。
- aclnnIncreFlashAttention/aclnnIncreFlashAttentionV2/aclnnIncreFlashAttentionV3接口废弃，替换为：aclnnIncreFlashAttentionV4。
- aclnnPromptFlashAttention/aclnnPromptFlashAttentionV2接口废弃，替换为：aclnnPromptFlashAttentionV3。
- aclnnMlaProlog/aclnnMlaPrologV2WeightNz接口废弃，替换为：aclnnMlaPrologV3WeightNz。
- aclnnMatmulAllReduceAddRmsNorm接口废弃，替换为：aclnnMatmulAllReduce和aclnnAddRmsNorm。
- aclnnQuantMatmulAllReduceAddRmsNorm接口废弃，替换为：aclnnQuantMatmulAllReduceV2和aclnnAddRmsNorm。
- aclnnWeightQuantMatmulAllReduceAddRmsNorm接口废弃，替换为：aclnnWeightQuantMatmulAllReduce和aclnnAddRmsNorm。
- aclnnInplaceQuantMatmulAllReduceAddRmsNorm接口废弃，替换为：aclnnQuantMatmulAllReduceV2和aclnnAddRmsNorm。
- aclnnInplaceMatmulAllReduceAddRmsNorm接口废弃，替换为：aclnnMatmulAllReduce和aclnnAddRmsNorm。
- aclnnInplaceWeightQuantMatmulAllReduceAddRmsNorm接口废弃，替换为：aclnnWeightQuantMatmulAllReduce和aclnnAddRmsNorm。

### 模型压缩工具

模型压缩工具以下特性标记为废弃，废弃的特性将在cann 9.0.0之后的版本删除。
- 非均匀量化
- 自动混合精度
- 近似校准
- int4量化感知训练
- amct_mindspore所有特性

## 已知问题

问题一：在通信域使用int64算子的模型中，发生断链时，快恢耗时有分钟级增加到十几分钟级，影响MTTR
【引入版本】CANN 9.0.0
【缺陷影响】int64算子不支持重执行流程，无法走STEP快恢，影响断联场景下任务重执行 
【规避方案】不使用int64类型的集合通信算子，使用其他类型的算子临时替代

问题二：训练任务恢复时sendrecv算子卡住，导致进程级恢复失败，影响MTTR
【引入版本】CANN 8.5.0
【缺陷影响】跨超场景下，sendrecv算子偶现卡死，快恢失败
【规避方案】如果是逻辑跨超场景下，可以走HCCS链路进行规避，物理跨超场景暂无规避方案

## 已修复问题

以下问题在9.0.0版本修复：
- 修复了catlass算子编译报错“ld.lld: error: undefined symbol: CheckLogLevel”的问题。
- 修复了在Atlas 900 A2硬件上，PadV3Grad 算子因不支持[16,16,16,16] 规格的 padding 输入，执行过程上报“EZ9999”错误码的问题。
- 修复了MutMul算子在矩阵乘(32162, 4096)@(32162,32768)运算时，计算结果出现NAN的问题。
- 修复了MIMO-V2模型在MindSpeed-LLM框架训练场景下，FA算子稳定出现NAN的问题。
- 修复了aclnnQuantMatmul算子，当输入参数为切片参数时，报错shape不一致的问题。
- 修复了FAG算子在(1,32,641,128) no mask场景下，因数据切块丢失尾部数据，导致模型调用超时的问题。
- 修复了一个集群任务中环境同时存在两个驱动版本，使用reduce通信算子时，出现HCCL通信卡死导致任务运行失败的问题。

## 文档变更说明

### 环境准备

《CANN 快速开始》调整至“环境准备”节点下，并变更为《CANN 快速安装》。

### 编程指南

#### Ascend C算子开发

- 快速入门新增SIMD、SIMT简介，新增基于SIMT编程的快速入门章节。
- 编程指南中编程模型新增“AI Core SIMT编程”、“SIMD与SIMT混合编程”章节。
- 编程指南中编译与运行新增“AI Core SIMT编译”章节。
- 编程指南中语言扩展层新增SIMT BuiltIn关键字、SIMT语言扩展层C API。
- 算子实践参考中新增SIMT算子实现、SIMD与SIMT混合算子实现章节。
- 算子实践参考中新增“SIMD与SIMT混合算子性能优化”章节。
- API参考中新增SIMT API章节。
- 新增兼容性迁移指南（220x架构版本迁移到351x架构版本）。
- 新增90+ Reg编程接口API，Reg矢量计算API是面向RegBase架构开发的API，用户可通过该API直接对芯片中涉及Vector计算的寄存器进行操作，实现更大的灵活性和更好的性能。
- 昇腾社区中，Ascend C算子开发新增可视化专区，通过视频呈现Cube和Vector算子的执行过程。

#### 通信算子开发

- 《通信算子开发》从《HCCL集合通信库》中独立出来，作为单本手册发布至“编程指南”节点下。
- 新增“AIV通信算子开发”章节。

#### 应用开发

《应用开发》文档大纲调整，按功能特性调整一级目录，调整点如下：
- 将“运行时管理”章节下的内容提升一个级别。
- 增补Runtime更多功能特性的开发指导，例如ACL Graph、多设备编程、进程间通信等。
- 将接口调用流程的内容下沉到各个具体功能特性的章节中。
- 单算子调用章节下的内容区分内置算子、自定义算子。
- 媒体数据处理、特征向量检索功能变更至“专用加速器”章节，同时在“专用加速器”章节下增加随机数生成功能的介绍。
- acl API参考按组件拆分，分成Runtime API、GE API、DVPP API、特征向量检索API，API参考拆分。

#### 图开发

- 手册大纲调整：内容进行整合与重构，开发态内容移入“编程指南”章节。
- 《AutoFuse自动融合用户手册》合并到《图开发》的“自动融合”章节。
- 《图开发》新增“附录>Session到GeSession的迁移指导”章节。
- 新增“编程指南>Triton入图”章节。

### API参考

#### GE图引擎 API

- 接口参考按照语言进行分类，分为C++语言、Python语言、C语言接口，其中Python语言接口为新增接口。
- C语言接口新增“模型管理和单算子调用接口”，该部分内容来自原有的acl接口。

#### HCCL集合通信库

- 《HCCL集合通信库》的“相关参考>集群信息配置”章节下，新增“rank table配置资源信息（Atlas 350 加速卡）”章节。
- 《HCCL集合通信库》的“通信算子开发”章节下，新增“AIV算子开发”章节。

### 开发工具

《msLeaks内存泄漏检测工具》命名变更为《内存分析工具》。

## 漏洞修补列表

版本开源及第三方软件漏洞修复情况详见[漏洞修补列表](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/maintenref/refdoc/refer002.html)。

## 子包独立升级

### 使用场景

子包独立升级提供了灵活的按需升级功能。

使用场景举例：某用户已安装了a.b.c版本的组合包Ascend-cann-toolkit和Ascend-cann-ops，由于业务需要，要求将单边通信库能力更新至x.y.z版本，此时，可通过独立升级子包cann-hixl来实现。其中a.b.c和x.y.z请替换为实际版本号。

### 操作步骤

升级子包前，请确保已安装配套版本的组合包Ascend-cann-toolkit和Ascend-cann-ops。如未安装，请参考《[CANN 软件安装](https://hiascend.com/document/redirect/CannCommunityInstSoftware)》进行安装。

子包的升级路径应与组合包的安装路径保持一致。所有子包的升级操作相同，以下以升级 cann-hixl\_x.y.z\_linux-aarch64.run 为例进行说明。

**升级场景一：**

**组合包**Ascend-cann-toolkit和Ascend-cann-ops**安装在默认路径下**（即安装时未使用“--install-path”参数）

```bash
chmod +x cann-hixl_x.y.z_linux-aarch64.run 
./cann-hixl_x.y.z_linux-aarch64.run --upgrade
```

**升级场景二：**

**组合包**Ascend-cann-toolkit和Ascend-cann-ops**安装在指定路径**“/home/custom\_path”（即安装时指定“--install-path=/home/custom\_path”）

```bash
chmod +x cann-hixl_x.y.z_linux-aarch64.run
./cann-hixl_x.y.z_linux-aarch64.run --upgrade --install-path=/home/custom_path
```
