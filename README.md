# CANN 9.0+、A2/A3 Ascend C 算子优化实战：18 篇源码案例

本合集聚焦使用 Ascend C 实现的算子内部优化：Tiling、数据搬运、片上缓存、Cube/Vector 流水、归约、量化与通信融合。每篇都附有实际内核源码、固定 commit、来源和复现说明。整理日期：2026-10-08。目录沿用原编号，保留上游 Host、构建文件和示例等必要配套文件。

其中 17 篇使用官方 `v9.0.0` 正式源码；另外 1 篇是有明确 CANN 9.0 兼容依据的 SGLang Dispatch/Combine，其设备内核使用 Ascend C。版本依据来自[官方 9.0.0 配套发布表](https://gitcode.com/cann/release-management/blob/c847330e202fbe9e724ef36c6219e49f42cda312/9.0.0/release-notes.md)与 [SGLang DeepEP 配套说明](https://github.com/sgl-project/sgl-kernel-npu/blob/653e519cf9556b0cc8d9e7966aa4551eb65143ae/python/deep_ep/README.md)。

A2 对应本文的 910B 路径；A3 使用 `ascend910_93` / 910C 路径。每篇单独核对产品与源码分支，适用范围见下表。02 的正式快照 Host 仅接受 910B；21 在 A2 限训练产品；23 限 A2/A3 训练产品。A3 的准确 SoC 名称应以设备查询结果为准。

**性能证据必须与源码版本分开阅读。** 本机没有 Ascend NPU，未做 NPU 编译、精度或性能测试。公开历史时延的测量 CANN 版本多数未披露，因此不能把这些数字表述为已经复测的 CANN 9.0.0 成绩。详见各篇 `docs/sources.md`、`docs/metadata.json` 和[整包验证记录](VALIDATION.md)。

## 如何选择案例

下表中，“公开 A/B”有明确前后时延；“公开时延”只有参考绝对值；“源码”是生产内核的具体实现解析与可执行复现入口；“版本汇总”是官方版本级模板平均优化报告。18 篇分别对应不同内核或独立优化路径。

| # | 案例 / 文章 | 实际优化机制 | 设备范围 | 性能证据 |
|---|---|---|---|---|
| 01 | [DeepSeek LightningIndexer](01-deepseek-lightning-indexer/docs/post.md) | score GEMM、FixPipe ReLU、CV 双缓冲、局部排序与 TopK 归并 | A2/A3 | 源码 |
| 02 | [DeepSeek SparseFlashAttention](02-deepseek-sparse-flash-attention/docs/post.md) | 离散 KV Gather、DMA 配对、四槽缓存、三任务流水与在线 Softmax | A2 推理；A3 Host 证据不闭合 | 源码 |
| 03 | [MoE GMM + SwiGLU + Quant](03-moe-grouped-matmul-swiglu-quant/docs/post.md) | A8W8 NZ、专家 tile 调度、整行就绪、反量化与 RowMax 量化 | A2/A3 | 源码 |
| 04 | [DeepSeek MLA Prolog](04-deepseek-mla-prolog/docs/post.md) | Split-N/head、A 全载、权重预取、RMSNorm/RoPE/Cache 融合 | A2/A3 | 源码 |
| 05 | [MoE Dispatch / Combine](05-moe-dispatch-combine/docs/post.md) | 路由、INT8 分发、RDMA/HCCS、带权合并 | A2/A3；公开时延为 A3 | 公开时延 |
| 08 | [RmsNorm](08-rmsnorm/docs/post.md) | 短列多行合并、gamma 复用、长列 Split-D 两遍归约 | A2/A3 | 源码 |
| 09 | [AddRmsNormQuant](09-add-rmsnorm-quant/docs/post.md) | 残差、归一化与量化融合；按模式计算 UB 缓冲预算 | A2/A3 | 源码 |
| 10 | [LayerNormV4](10-layernorm/docs/post.md) | Single Read、释放 x 后复用存储、短行转置归约 | A2/A3 | 源码 |
| 11 | [AddLayerNorm](11-add-layernorm/docs/post.md) | 整行少缓冲、延迟参数加载、短行特殊归约 | A2/A3 | 源码 |
| 12 | [FP32 Matmul AL1 全载](12-matmul/docs/post.md) | 小 M 大 K 的真实选择条件、ND→L1 NZ、A 重用 | A2/A3 | 源码 |
| 13 | [WeightQuantBatchMatmul](13-weight-quant-batch-matmul/docs/post.md) | NZ 权重反量化、Split-K、四槽 GM/CV 流水与原子累加 | A2/A3 | 源码 |
| 14 | [QuantBatchMatmul per-token](14-quant-batch-matmul/docs/post.md) | scale 外积预计算、INT32 workspace、分组同步与背压 | A2/A3 | 源码 |
| 15 | [DeepSeek 尺寸 DynamicQuant](15-dynamic-quant/docs/post.md) | 专用 shape Tiling、多行 smooth 预算、超长行两遍扫描 | A2/A3 | 源码 |
| 16 | [RoPE](16-rope/docs/post.md) | 广播 cos/sin 重用、half/interleaved 旋转、局部融合乘加 | A2/A3 | 源码 |
| 20 | [SwiGLU](20-swiglu/docs/post.md) | 跨行 DMA、FP32 临时空间复用、按 tile 数切换单/双缓冲 | A2/A3 | 源码 |
| 21 | [Fused Floyd Attention](21-fused-floyd-attention/docs/post.md) | 两路 QK/PV 融合、在线累积、三任务/双结果 CV 环 | A2 训练；A3 训练/推理 | 公开 A/B |
| 22 | [DeformableConv2d](22-deformable-conv2d/docs/post.md) | 坐标复用、小 C 相邻访存合并、插值重排与 Cube | A2/A3 | 源码 |
| 23 | [FlashAttention 反向](23-flash-attention-grad/docs/post.md) | SAMEAB/BN2 基础 Cube API、L1 输入复用、CV 流水及确定性归约 | A2/A3 训练 | 官方版本汇总 |

优先研究有前后时延对照的实现可从 21 开始；DeepSeek 内核阅读顺序为 01→02→04→03→15→05。矩阵乘的数据流可对照 12、13、14；归一化的 shape 分流与 UB 预算可对照 08–11；训练反向与复杂访存可看 23、22。每篇末尾给出应记录的 shape、dtype、Tiling、workspace、精度和 profiling 指标。

## 文件结构与复现

```text
blog/
├── README.md                  # 人工阅读目录
├── catalog.json               # 统一机器可读索引
├── SELECTION.md               # 纳入标准与排除理由
├── VALIDATION.md              # 本机验证范围与结果
├── validate_bundle.py         # 离线检查工具
├── bundle-manifest.json       # 整包 SHA256（不含清单自身）
└── 01-… / … / 23-… /          # 共 18 个 post，保留原编号
    ├── code/
    │   ├── benchmark.py       # 仅 01、02 两篇有
    │   └── upstream/          # 原始内核、Host、示例及许可证快照
    └── docs/
        ├── post.md            # 具体路径、机制、成本与边界
        ├── reproduce.md       # 环境、构建、精度、性能测量
        ├── sources.md         # 固定版本的一手依据
        └── metadata.json      # 版本、硬件和测量状态
```

`upstream/` 是选取的阅读快照。`docs/reproduce.md` 保留供目标设备参考的手动命令，`docs/sources.md` 给出固定版本与核心源码来源。本文整理过程仅涉及文件与离线校验，未在当前电脑执行这些设备端命令。

在 blog 目录运行离线整包检查：

```bash
python3 validate_bundle.py
```

CANN 9.1 及更高版本应使用对应配套发布源码并重新编译、测精度和性能。本合集给出可追踪的 9.0 基线，未承诺同一二进制跨所有后续版本兼容。

原始文件保留其许可证与版权声明。CANN 源码使用 CANN Open Software License Agreement Version 2.0；SGLang 快照保留 MIT License。详见每篇 `code/upstream/LICENSE`。
