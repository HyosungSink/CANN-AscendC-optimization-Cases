# QuantMatmulReluQuant 实现记录

> 文档快照：commit `1e3f129`（2026-09-27）。本页前半部分描述当前有效版本，后半部分保留按时间倒序的实验记录。标注“已回退”的实验不属于当前实现。

## 当前有效方案摘要

### 目标和数值契约

算子固定处理 INT8 输入：`x1[M,K]` 与按 `x2[N,K]` 存储的右矩阵，计算
`acc = x1 @ x2.T`，随后按行/列 scale 做 FP32 反量化、ReLU、逐行动态量化，输出
INT8 `y[M,N]` 和 FP32 `yScale[M]`。`y` 必须 bit-exact，量化使用 round-to-even；
near-half 修正和 `CAST_ODD` 不能在没有 golden 对拍的情况下删除。当前工程没有声称
支持其他输入数据类型，线上真实形状也未知。

### 已确认环境

| 项目 | 当前值 |
|---|---|
| NPU | Ascend910_9362 |
| 计算资源 | 20 AI Core / 20 Cube / 40 Vector |
| 片上存储 | UB 192 KiB，L1 512 KiB |
| CANN | 9.0.0（`/home/developer/Ascend/cann-9.0.0`） |
| 编译架构 | `dav-2201` |
| 当前源码 | `kernel.asc`，host 入口 `main.asc` |

硬件信息通过 `npu-smi`、`aclrtGetSocName()`、`aclrtGetDeviceInfo()` 和 CANN
platform 配置交叉确认；不要仅依据 `npu-smi` 中的通用 `Ascend910` 名称推断型号。

### 当前 host 调度

`run_kernel` 按以下顺序选择路径，前面的条件优先：

1. `sourceK > 65535`：`HugeKKernel`，使用八路 INT32 AIV 累加，绕过 Matmul tiling
   字段边界，覆盖超大 K 的正确性场景。
2. 小 M、短 K、非对齐 N/K：`SmallKernel`、`SmallBatch2Kernel`；物理面板按 32/64/128
   对齐，实际读写仍使用 `sourceN/sourceK`，避免尾部越界。
3. `K=128` 且 `M<=128,N<=128` 的非标准列宽：使用 padded Vector 面板；常见
   `N=16/32/64` 且 `M=97..128` 使用三行批处理复用 B。
4. 其他小矩阵：对 `K=32/64/128` 的 48/80/96 等列宽使用 Vector 特化；剩余尺寸
   进入 Cube tiling。
5. 通用 Cube 路径使用 `FusedKernel`；短行宽 N 使用 `WideKernel`，特定大 N/大 K
   使用 `ColumnStreamKernel` 或 `LongColumnStreamKernel`，由 AIC 生产、AIV 消费列组。
6. 当 `N>=16384`、`sourceK<=256` 且 K 非 32 对齐时，先用 `PadInputsKernel` 将 A/B
   补齐；workspace 尾部放置 padded 输入。当前方案明确允许 host 侧 `aclrtMalloc`，
   因此不能再引用旧的“无 workspace”架构描述。

Cube 路径将 INT32 accumulator 写入 host 分配的 workspace，AIV 完成反量化、ReLU、
行最大值和量化；cross-core flag 保护 accumulator slot 的复用。所有尾块按真实 M/N
写回，非对齐输出必须继续通过 guard 和 golden 检查。

### 最新本地验证快照

以下数据来自当前源码最近一次完整回归；本地估分用于比较版本，不等同于线上隐藏
形状的得分：

| 验证项 | 结果 |
|---|---|
| 主测试 | 15/15 PASS，估算均分约 **20.592** |
| patterns | 12/12 PASS |
| boundaries | 22/22 PASS |
| 主测试耗时（us） | `18.62, 14.62, 14.96, 14.60, 55.82, 91.88, 72.60, 72.74, 130.74, 476.98, 492.54, 992.14, 747.46, 1025.12, 1610.90` |
| 重点边界 | `K=65535/65536/65537`、`N=1`、非对齐 M/N/K、宽 N 尾块、超大 K |

### 快速复现

```bash
cmake --build build -j4
cd build
python3 ../scripts/test_dimensions.py
python3 ../scripts/test_dimensions.py --patterns
python3 ../scripts/test_dimensions.py --boundaries
python3 ../scripts/test_dimensions.py --shape 17 2049 257
```

单点计时使用设备 event；小矩阵路径在 kernel 内异步返回，端到端计时必须由调用方
统一同步，不能把未同步的 host 返回时间与通用路径的同步时间直接比较。需要 profile
时，在 `build/` 中先生成匹配输入，再执行：

```bash
msprof --output=profile_case15 --aic-metrics=PipeUtilization \
  ./quant_matmul_relu_quant_custom 6784 3509 2196
```

当前环境能够稳定得到 kernel duration，但 `msprof` 偶尔因 File I/O/权限限制不导出
完整 summary；缺少 summary 时只记录 kernel 时间，不推断 Cube、Vector 或 MTE 利用率。

### 当前已知限制

- `K>65535` 的 `HugeKKernel` 以正确性为先，吞吐明显低于 Cube 路径。
- `WideKernel` 解决的是 Cube 侧短行并行；`M` 极小时 AIV 后处理仍可能成为串行瓶颈。
- `PadInputsKernel`、accumulator 和 padded A/B 共用一次 host workspace，生命周期必须
  覆盖 stream 上的设备使用；不能在异步调用后立即释放。
- 线上实际 15 个形状不可见，任何本地均分或专项 profile 都不能宣称为线上成绩。

## 历史实验日志（按时间倒序）

### 2026-09-27 用户提供 kernel 架构重测

按用户给出的完整 AscendC 实现替换 `kernel.asc`，保留其 `PadInputsKernel`、
Cube/AIV 流水、`ColumnStreamKernel`、`WideKernel` 以及小尺寸专用 kernel。
硬件为 Ascend910_9362（20 Cube/40 Vector，UB 192 KiB，L1 512 KiB），
CANN 9.0.0，编译架构 `dav-2201`。`cmake --build build -j4` 编译通过，
仅有 `GM_ADDR` 强转的编译器警告。

新实现原样在 `K=65536` 处出现错误（`K=65535` 正常），这是 Matmul tiling
字段边界；为保持任意正整数 K 的精度，在 `sourceK > 65535` 时加入旧版已验证
的 AIV 八路 INT32 累加兜底。专项 `1x3x65536`、`1x3x65537`、
`1x3x135169` 均恢复 PASS。该实现使用 `aclrtMalloc/aclrtFree` 申请临时
accumulator，且 `padK` 时在同一分配尾部放置 padded A/B，和“无 workspace”
版本相比属于明确的额外 workspace。

验证命令及结果：

- `python3 ../scripts/test_dimensions.py`：15/15 PASS，耗时（us）
  `59.50, 56.08, 67.50, 14.60, 55.42, 92.18, 72.70, 67.70, 138.36,
  507.52, 570.44, 1027.60, 790.28, 1075.42, 1552.20`，估算均分 **19.186**。
- `python3 ../scripts/test_dimensions.py --patterns`：12/12 PASS。
- `python3 ../scripts/test_dimensions.py --boundaries`：22/22 PASS，包含
  `K=65535/65536/65537` 和非对齐尾块。
- `msprof --output=profile_user_kernel_case15 --aic-metrics=PipeUtilization
  ./quant_matmul_relu_quant_custom 6784 3509 2196`：kernel 1576.46 us；
  当前环境未导出 summary，不能据此报告 Cube/Vector/MTE 利用率。

相对替换前约 13.7 的本地均分，本轮约 **19.186**；性能收益来自 Matmul
Cube 路径、宽矩阵列并行和批量 epilogue。超大 K 兜底仍是标量 AIV 路径，
对应尺寸性能较低但精度正确。

### 2026-09-27 大 N 列流式扩展实验（保留）

依据代码审查中“列流式路径可以处理尾部 512 列组”的建议，先尝试修正普通
`ProcessRows` 的连续行均分。该改动在 `192x2048x1024` 上由 71.70 us 退化到
86.76 us，在 `192x4096x1024` 上由 88.30 us 退化到 113.82 us，说明它破坏了
当前 batch/双 Vector 子核的流水布局，已回退。

随后只扩展 `ColumnStreamKernel`：列组改为 ceil 分组，最后不足 512 列的组使用
真实宽度，并将保留的完成通知从 flag 8 移到 flag 15；调度条件从
`N<=4096,N%1024==0` 放宽为 `1024<=N<=8192`。AIC 对尾组发送 Fix 通知，AIV
按实际宽度处理，未改变数值计算顺序。

专项结果：`576x5760x4500` 约 570.44 -> 510.02 us，
`320x7424x7008` 约 1027.60 -> 1005.90 us，
`768x6656x5056` 约 1075.42 -> 1016.94 us；所有专项 PASS。

完整验证：编译通过；15/15 主测试、12/12 patterns、22/22 boundaries 全部 PASS。
本轮 15 点实测耗时为
`44.44,45.98,52.96,14.28,58.26,89.70,72.74,66.48,125.96,490.78,
502.24,1011.28,766.40,1022.62,1578.80 us`，估算均分 **19.582**，相对上一版本
19.186 有约 2.1% 均分提升。`N=5760/7424` 的尾列已由专项和 boundaries 对拍覆盖。

结论：保留列流式尾组扩展；不保留普通 `ProcessRows` 均分实验。flag 15 是该路径
专用完成通知，后续修改列组数量时必须保持 group flag 与完成 flag 不冲突。
本轮 `msprof --output=profile_column_tail_case11 --aic-metrics=PipeUtilization
./quant_matmul_relu_quant_custom 576 5760 4500` 采集因环境 `Operation not permitted`
失败，未得到可引用的 Cube/Vector/MTE summary。

### 2026-09-27 tie 候选门控（保留）

硬件仍为 Ascend910_9362（20 Cube/40 Vector、UB 192 KiB、L1 512 KiB），
CANN 9.0.0、`dav-2201`。恢复原 near-half 修正后，15/15 PASS，估算均分
13.226；第 13 点 `8704x2049x32` 为 2898.28 us。

理论：随机数据中 `abs(CAST_RINT(raw)-raw)>0.49999` 的候选极少，原路径
每个 N tile 都生成 mask 并用 Scalar 扫描所有 mask 字。改为先在 UB 对残差
做 `ReduceMax`；未越阈值的 tile 跳过 mask 和扫描，候选 tile 保留原逐值
FP32 顺序及 half-to-even 修正，不改 rounding 语义。

`msprof --output=profile_before_gate_case13_valid --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 8704 2049 32`
与 `profile_after_gate_case13` 对照：task 约 2928.04 -> 2487.05 us，
AIV scalar 1918.68 -> 1432.90 us、Vector 1066.52 -> 1113.70 us、
AIC MTE2 100.95 -> 89.15 us。第 13 点 ACL event 2898.28 -> 2469.88 us。
完整 15/15、12/12 模式、22/22 边界 PASS；15 点估算均分 13.323。
第 15 点约 10381.86 -> 10048.24 us。结论：保留；收益来自减少
Scalar mask 扫描，但 Cube 与宽 N 的 GM 搬运瓶颈仍在。

下一实验理论：pass0 当前每个 N tile 做 16 次 `ReduceMax` 后立即
Vector->Scalar 等待并逐行读回，造成大量短同步。尝试将 rowMax 留在 UB
以 `Max` 向量累积，整行结束时才读回；这只改变最大值归约的保存位置，
不改 activation 的 FP32 乘法顺序。

### 2026-09-27 UB rowMax 向量累计（保留）

理论：原 pass0 每个 N tile 的 16 个 rowMax 都执行一次 Vector->Scalar
同步和 `GetValue`。改为 `ReduceMax` 后在 UB 用向量 `Max` 跨列 tile
累计，到整行完成时才同步并计算 `yScale`，避免每 tile 的标量归约。
`msprof --output=profile_after_ub_rowmax_case13 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 8704 2049 32`
对照门控版本：第 13 点 task 2487.05 -> 2364.59 us，AIV scalar
1432.90 -> 1285.11 us、Vector 1113.70 -> 1140.31 us，AIC MTE2
约 89.15 -> 90.80 us。完整 15/15、12/12 模式、22/22 边界 PASS，
均分 13.406；第 13 点 2469.88 -> 2350.98 us，第 15 点约 10.05 ms
基本持平。结论：保留，主要改善短 K 宽 N 的 AIV 开销。

下一实验理论：当前每个 M tile 的 `x1Scale` 对每个 N tile 重复 GM
标量读取；提前读入 16 元素本地数组可减少这类访问，仍保持原逐元素
FP32 乘法顺序。必须以完整 15 点和 profile 评估，不能仅凭第 13 点保留。

### 行 scale 预取（已回退）

仅将每个 M tile 的 16 个 `x1Scale` 标量预取到本地数组，替代每个 N tile
的重复 GM 读取。第 7/13/15 点均 PASS，但耗时约 357.82/2361.78/10060.34 us，
对照上一版本 357.58/2350.98/10053.94 us 无收益。
`msprof --output=profile_lhsscale_hoist_case13 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 8704 2049 32`
显示 task 2364.59 -> 2387.11 us，AIV scalar 1285.11 -> 1308.20 us。
推断本地数组的动态索引抵消了减少 GM scalar load 的收益；已回退，
没有因为理论上的访问减少而保留无效代码。

### 非对齐输出的 64 位 UB staging（已回退）

`OutputWriter` 在源地址不是 32 字节对齐时逐字节 `GetValue/SetValue`。
对源偏移及长度 8 字节对齐的片段尝试按 `uint64_t` 复制到 staging，
最终 GM 写回和原有 MTE3 同步不变；非 8 字节对齐片段保留逐字节路径。
理论收益集中在奇数 N 的短尾 tile，例如第 13 点的 16 列尾块；需
通过 M/N/K 非对齐、scratch 前缀恢复和所有 golden 验证。
专项 `8704x2049x32`、`33x16401x32`、`65x543x257`、`17x31x33`
均 PASS；`msprof --output=profile_packed_staging_case13 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 8704 2049 32`
显示第 13 点 task 2364.59 -> 2350.99 us、AIV scalar 1285.11 ->
1260.92 us。但完整 15 点均分 13.391，低于原 13.406；第 13 点
2350.98 -> 2331.80 us，其余点整体无稳定收益。新增分支不符合
简化目标，因此回退。未改 rounding 或 scratch 生命周期。

### 小尺寸 AIV fallback 的 UB 输入驻留（已回退）

case4 `8x16x64` 的 `msprof --output=profile_before_ub_fallback_case4 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 8 16 64`
显示 task 30.44 us、AIV scalar 27.41 us（90.0%），Vec/MTE 几乎为零。
现有 fallback 在每个 `(row,col)` 的 K 循环逐元素访问 GM A/B，B 被
每行重复读取。实验将 B 总量不超过 8 KiB、K 不超过 128 的 fallback
输入装入 UB，每行 A 装入 UB，其余尺寸保持原路径；仍使用 INT64 分路
累加后转 INT32，不改变后续 FP32 计算和半偶舍入。新增的 MTE2 同步
可能抵消收益，需以 case1/3/4、非对齐边界和完整均分判断。
case1/2/3/4 均 PASS，但 case1/3/4 约 98.66/57.84/82.34 us，
相比原 38.72/约 33-44/约 33-44 us 明显退步。
`msprof --output=profile_ub_fallback_case4 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 8 16 64`
显示 case4 task 30.44 -> 79.64 us，AIV scalar 27.41 -> 74.71 us、
MTE2 约 0 -> 2.02 us。UB 的逐元素 `GetValue` 比 GM scalar 路径更慢，
只换存储位置不够；已回退。未来若处理小尺寸，需要向量化点积，
而非继续添加 Scalar/UB 版本。

### 2026-09-27 最终保留状态（历史快照）

最终保留 tie 候选门控与 UB rowMax 向量累计；行 scale 预取、64 位 staging、
小尺寸 UB 输入驻留均已回退。重新编译后的完整 15 点命令：
`cd build && python3 ../scripts/test_dimensions.py`，15/15 PASS，估算均分
**13.400**。最终关键耗时（us）为：

`38.74, 14.24, 33.62, 34.06, 77.66, 113.82, 357.20, 435.54,
442.14, 861.24, 3309.96, 3235.82, 2344.36, 5716.30, 10077.88`。

关键专项和此前已验证的 12 个模式、22 个边界仍通过；最终重新编译后又复核
`8704x2049x32`、`576x5760x4500`、`6784x3509x2196`、`1x3x65537`，
均 PASS。`git diff --check` 无错误。当前主要剩余瓶颈由 msprof 确认：
宽 N/大 K 的 AIC MTE2、MTE1 和 AIV 输出整理；若继续冲击 20 分，应重构
M tile 间 B tile 生命周期或用真正向量化的 fallback 点积，不能继续堆标量
缓存分支。

### 2026-09-27 线上 WA 回退

最新线上反馈在测试点 6、9、10、12、13 出现 0.05%--0.20% 错误，
而本地随机、模式、边界均 PASS。逐项审查后，最近两项改动存在硬件相关风险：

- tie 门控使用 `ReduceMax` 判断是否生成 mask，可能在特定尾 tile 上跳过应有的
  near-half 修正；
- UB `rowMaxLocal` 使用原地向量 `Max` 累计，改变了原先逐 tile 的归约同步。

已将两者都恢复为原始、已验证的路径：每 tile `ReduceMax` 后 Scalar 更新
`rowMax[]`，每个量化 tile 始终生成 tie mask 并扫描非零候选。重新编译后
12 个模式、15 个标定点以及额外的非对齐和大 M 尺寸均 PASS；本地均分约
13.03，性能回退是有意的 correctness-first 取舍。下一次线上提交应先确认
WA 消失，再考虑只在明确安全的整齐 tile 上重新启用门控。

### 2026-09-27 官方流水建议：B1 ping-pong（保留）

参考官方 DoubleBuffer 文档和流水优化文章：MTE2/MTE1/M/Vector 属于可并行
指令队列，双缓冲必须让下一块搬运在当前计算期间发射，不能只增加 buffer
或 event。官方同时提醒小数据和计算占主导时强行 DoubleBuffer 可能退化。
资料整理见 [OFFICIAL_ASCEND_OPTIMIZATION_NOTES.md](https://gitcode.com/liu-jingzong/cann-1/blob/12f392638497f3f1442f7c46407ef96588cc31f2/OFFICIAL_ASCEND_OPTIMIZATION_NOTES.md)。

当前 C220 手写路径的瓶颈是每个 K tile 都在覆盖同一个 B1 前等待
`MTE1_MTE2`。本实验保留 `M_TILE=16/N_TILE=128/K_TILE=384` 和原 NZ
布局，只将非 `cacheB` 路径的 B1 改为两个 49,152B slot（每 slot
`128 * align32(384)`），每个 slot 使用独立 `MTE1_MTE2` event：

```text
load B(slot 0) -> load L0B -> MMAD(K0)
load B(slot 1) -> load L0B -> MMAD(K1)
load B(slot 0) 与 MMAD(K1) 重叠，覆盖前等待 slot 0 event
```

`cacheB` 的 K<=64 预装路径未改变；尾 K、尾 N 和 AIC/AIV scratch 握手未改变。
针对 `55x55x105`、`64x7808x1024`、`576x5760x4500`、`8704x2049x32`、
`6784x3509x2196` 的 golden 全部 PASS。

结果：

- 完整 15/15 PASS，12/12 patterns PASS，22/22 boundaries PASS；
- 本地估算均分 **13.737**（回退版约 13.03）；
- 第 10/11/12/14/15 点：约 `612.62/2244.42/2129.94/3541.10/7604.58 us`；
- 第 15 点相对回退版约 10.38 ms 降至 7.60 ms，证明减少 B1 覆盖等待有效；
- 小尺寸和 K=32 的第 13 点基本不变，符合只优化 K-loop 的理论。

命令：

`msprof --output=profile_b1_pingpong_case15 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 6784 3509 2196`

该次 msprof 由于环境 File I/O 限制未导出完整 `op_summary`，但 ACL kernel
时间和 15 点结果稳定；下一轮应在可导出 profile 的环境重新采集 AIC
MTE2/MTE1/MAC，确认重叠比例。不要把两个 B1 slot 扩到已触发 507015 的
`N_TILE=144/160` 或 `K_TILE=416`。

### 2026-09-27 宽 N 局部 activation 缓存实验

硬件复核：Ascend910_9362，20 Cube/40 Vector，UB 192 KiB、L1 512 KiB，
CANN 9.0.0，编译 `dav-2201`。沿用同一工作区，不申请独立 workspace。
修改前 `cd build && python3 ../scripts/test_dimensions.py 15`：
`6784x3509x2196` PASS，13953.18 us。
`msprof --output=profile_before_partialcache_case15 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 6784 3509 2196`
的 task 14332.99 us、AIC MTE2 9262.38 us、MTE1 3323.39 us、MAC 661.19 us。
实验假设：原本宽 N 的 UB activation 缓存采取全有或全无，N=3509 因超过
2048 列容量而整行第二遍 GEMM；改成只缓存前 16 个 128 列 tile（2048 列），
第二遍仅重算剩余 tile。需保证双槽 `produced/consumed` 只计实际 GEMM tile，
并对齐 N=2048/2049、奇数 N、M 尾块和 ties 的 golden。

局部缓存初测：`6784x3509x2196` PASS，10364.08 us。修改后同尺寸
`msprof --output=profile_after_partialcache_case15 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 6784 3509 2196`
测得 task 10584.03 us、AIC MTE2 6653.42 us、MTE1 2357.60 us、MAC 470.01 us；
相对修改前 task -26.2%、MTE2 -28.2%，符合减少第二遍 GEMM 的预期。
完整 15 点局部缓存版本 PASS；第 10/11/12/14/15 点约为
860.16/3292.52/3238.52/5738.10/10410.50 us。12 模式、22 边界也 PASS。

额外实验：只使用硬件 `CAST_RINT`，关闭 near-half 标量修正。12 个模式包括
ties/below/above 全部 PASS，但完整 15 点第 11 点 `576x5760x4500`
在 `[62,4534]` 发现 1 个 `y` bit 差异。结论：修正不可直接删除，已恢复；
保持原 FP32 顺序与 half-to-even 语义。

### 2026-09-26 持续实验日志（历史）

每轮只保留 15 点正确且均分有收益的修改；使用同一个工作区，不申请独立 workspace。
硬件复核：`npu-smi info` 显示 Ascend910；项目 CANN platform_config 和
`aclrtGetSocName()` 为 `Ascend910_9362`，20 Cube/40 Vector、UB 192 KiB、L1 512 KiB；
CANN 9.0.0，CMake `dav-2201`。本轮起点 15/15 PASS、估算均分 **12.778**。

### 实验：N_TILE=256、K_TILE=192、UB cache=16x1280（已回退）

理论：B2 保持 48 KiB，宽 N 列 tile 与跨核握手约减半，UB activation cache 80 KiB。
命令：`cmake --build build -j4`；`cd build && python3 ../scripts/test_dimensions.py`；
`python3 ../scripts/test_dimensions.py --shape 64 6912 1024 7168 2049 32 17 2049 257`；
`msprof --output=profile_n256k192_case13 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 7168 2049 32`。
15/15 PASS，均分 **12.354**；第 8/13/15 点为 512.88/5118.86/21504.56 us。
第 13 点 task 约 5171 us，AIC MAC 18.9 us、MTE2 1422 us、AIV Vector 686 us、
AIV scalar 3964 us、MTE3 235 us。结论：宽 tile 未解决 AIV 标量输出开销，整体退步，
恢复 N_TILE=128、K_TILE=384、cache=16x2048。下一实验针对 scratch 前缀的标量暂存。

### 实验：scratch 前缀 UB 按行 32 字节对齐（保留）

理论：奇数 N 时原始线性偏移使整行落在非对齐 UB 地址，前缀 INT8 输出被逐字节
`SetValue`；改为每行按 32 字节对齐后，完整 N tile 可使用 UB 内 `DataCopy`。
15/15 PASS，均分 **12.862**；第 8/13/15 点 367.84/3131.94/17448.98 us。
12 个模式、22 个边界 PASS，包括 half-to-even、负 scale、K 极限和奇数 N。
命令：`python3 ../scripts/test_dimensions.py`、`--patterns`、`--boundaries`；
`msprof --output=profile_n128k384_case13 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 7168 2049 32`；
`msprof --output=profile_aligned_deferred_case13 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 7168 2049 32`。
第 13 点 task 4382.49 -> 3101.62 us，AIV scalar 3097.62 -> 1875.31 us；
AIC MAC 均为 32.61 us，AIC MTE2 1378.06 -> 1347.18 us。
结论：输出前缀的标量暂存是实际瓶颈；下一轮尝试把按行恢复合并成 2D MTE3。

### 实验：2D MTE3 恢复（已回退）

第一次尝试以 `blockLen=N`、`srcStride=align32(N)-N` 搬运，第 13 点出现
968077 个 `y` 差异，第二行开始错误；说明 UB 源步长很可能已按 32 字节自动进位。
改为 `srcStride=0` 后，奇数 N 专项 PASS，说明源步长已自动对齐。
然而第 7/8/9/13/15 点约为 283.06/367.76/524.34/3097.78/17658.14 us，
相对逐行恢复没有稳定收益；第 15 点从 17448.98 增至 17658.14 us。
结论：恢复次数不是主要瓶颈，回退至已通过 15 点、模式和边界测试的逐行恢复。

### 实验：K<=64 的 B 矩阵预装 L1（保留）

理论：第 13 点 K=32、N=2049，B 原始数据约 66 KiB，却随每个 16 行 tile
和两遍 GEMM 重复从 GM 搬运；基线 AIC MTE2 约 1347 us。若所有 NZ 列 tile
不超过 160 KiB，则每个 core 只预装一次 B 到 L1，后续直接 LoadData 到 L0B。
条件基于实际 `columnTiles * tileWidth * align32(K)` 字节数，其他尺寸不变。
15/15、12 模式、22 边界全部 PASS；均分 **12.908**，第 13 点 2784.50 us。
`msprof --output=profile_b_l1_case13 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 7168 2049 32`
测得 task 2809.92 us、AIC MTE2 94.58 us，对照修改前 3101.62/1347.18 us；
AIV scalar 1755.66 us，故总耗时仍受 AIV 和跨核等待限制。其余大 K 点无明显回归，保留。

### 实验：UB activation cache 从 16x2048 扩到 16x2176（已回退）

理论：N=2049 经 128 列 tile 补齐需要存 16x2176 个 FP32，当前差 8 KiB，
使 AIV 和 Cube 重跑第二遍。静态 UB 分配估算由 185856 增至 194048 字节，
低于 192 KiB 上限 196608 字节；需实测资源和流水停顿，不能只按 GEMM 次数判断。
专项均 PASS，但第 13 点 2784.50 -> 2987.58 us、`64x2050x33`
约 268.78 -> 298.94 us。缓存省掉第二次 GEMM，却令单槽交接等待 AIV 量化，
损失了双槽流水；恢复 16x2048。`64x2176x32` 为 166.02 us，
`64x2177x32` 为 513.14 us，说明容量边界对小 M 仍重要，后续需要同时重构
缓存与 pipeline 才有收益。

### 实验：将 B L1 预装扩展到单 K tile（已回退）

由原 `K<=64` 改为 `K<=K_TILE`，同时继续要求所有列 tile 的 NZ B 不超过
160 KiB。理论上单 K tile 不需要 K 分块布局变换，覆盖第 5 点等小矩阵，
避免重复 GM B 读取；重点检查预装成本是否影响只有一个 row tile 的情况。
15/15 PASS，但均分 **12.898**，未超过 K<=64 的 **12.908**；
第 5 点 78.86 -> 76.64 us，其余起伏与测试波动同量级。
按一轮一变量原则回退条件，不为微小局部收益增加路径覆盖和验证负担。

### 当前保留版本与下一轮方向

当前源码回到已验证的 `K<=64` L1 预装 + 对齐 UB 暂存，15/15 PASS，
12 模式、22 边界 PASS，估算均分 **12.908**。完整 15 点命令：
`cd build && python3 ../scripts/test_dimensions.py`。逐点 `(M,N,K): us`：

| 点 | M,N,K | us |
|---:|---|---:|
| 1 | 8,16,80 | 38.88 |
| 2 | 3,8,16 | 14.54 |
| 3 | 16,16,16 | 33.66 |
| 4 | 8,16,64 | 33.28 |
| 5 | 55,55,105 | 78.86 |
| 6 | 384,512,1248 | 137.64 |
| 7 | 64,5248,1024 | 284.26 |
| 8 | 64,6912,1024 | 366.66 |
| 9 | 1024,1248,960 | 541.08 |
| 10 | 320,4480,3841 | 1345.52 |
| 11 | 576,5760,7008 | 5756.72 |
| 12 | 320,7424,7008 | 3747.96 |
| 13 | 7168,2049,32 | 2784.50 |
| 14 | 768,6656,7008 | 9921.74 |
| 15 | 8279,3509,2196 | 17411.22 |

第 15 点命令：
`msprof --output=profile_current_case15_pipe --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 8279 3509 2196`。
task 17764.86 us，AIC MTE2 11268.44 us、MTE1 4060.18 us、MAC 807.78 us；
AIC scalar 15398.95 us 包含大量等待，非纯 scalar 运算量；AIV Vector 2172.73 us。
推断：当前 rowBase 外层循环令同一行组内多个 row tile 及第二遍重复加载 B。
下一轮应单独实验以 `N tile -> row tile` 为核心的 B1 复用，并维护每个 row tile 的
独立 rowMax、scratch 槽及交接顺序。以第 15 点为例，B 原始数据约 7.7 MiB，
不可能整体驻留 160 KiB B1；单列 tile 的 K 分块可驻留，需评估 B 少读与 A 重读
之间的权衡。此方案尚未实现或验证，不能记作已取得收益。

### 2026-09-26 无 workspace 统一架构（已被当前方案替代）

用户要求禁止独立 workspace；保持 14 参数 `run_kernel` 和每次只发射一个 kernel。
下方旧基线和待 workspace 决策的历史记录已被本节替代。实机仍为 `Ascend910_9362`，
20 Cube/40 Vector，UB 192 KiB、L1 512 KiB，CANN 9.0.0，`dav-2201`。
架构和资源限制详见 [UNIFIED_KERNEL_DESIGN.md](UNIFIED_KERNEL_DESIGN.md)。

### 代码与精度

- `kernel.asc` 从原工作区约 1993 行精简到约 635 行，仅保留一个主 Cube+AIV kernel、
  一个空间不足或 `K>12288` 时的 AIV 回退 kernel。
- 删除 6 个未调用旧 kernel、`phase`、`nChunks`、`partialGlobal`、`scratchGlobal1/2/3`、
  `StorePackedBytes`、所有 `SyncAll` 与主路径的 scalar M/N/K tail GEMM。
- 主路径 16x128x256，M 行组独占一个 `y` 前缀 INT32 中转槽；
  AIV 用 UB 保留该区最终 INT8，再一次恢复。受 C220 Fixpipe 通路及禁止 workspace 限制，
  这一小段 `y` 借用仍然存在，不能声称完全改成片上 buffer。
- N 较小时 FP32 activation 放 UB，避免第二次 GEMM 和重复 GM->UB INT32 搬运。
  N 超过 UB 容量时重算一次 GEMM；这部分仍待进一步架构优化。
- `main.asc` 的输出两端各增加 64 字节 guard；15 个标定点、12 个输入模式、
  22 个边界尺寸均经 NumPy golden 精度对拍和 guard 检查。模式包含 half-to-even 及
  相邻 FP32 值、负 scale、全零、INT8 极值、大累加值；边界包含 M=1、N=16401、
  N=2048/2049、K=12288/12289 和 K=65535/65536/65537。AIV 回退的部分和使用
  INT64 避免 C++ 有符号溢出，`K=135169` 且全 127 的 INT32 回绕专项通过。

### 最终 15 点（本地 ACL event，us）

运行 `cd build && python3 ../scripts/test_dimensions.py`；每点 y bit-exact、yScale 容差通过。

| 点 | M,N,K | us | 结论 |
|---:|---|---:|---|
| 1 | 59,35,56 | 93.560 | PASS |
| 2 | 46,43,62 | 90.820 | PASS |
| 3 | 67,64,33 | 63.140 | PASS |
| 4 | 107,58,70 | 114.020 | PASS |
| 5 | 79,55,105 | 138.920 | PASS |
| 6 | 76,96,89 | 109.520 | PASS |
| 7 | 67,79,153 | 131.540 | PASS |
| 8 | 78,155,70 | 242.880 | PASS |
| 9 | 61,167,97 | 211.300 | PASS |
| 10 | 124,166,127 | 241.220 | PASS |
| 11 | 138,269,95 | 271.020 | PASS |
| 12 | 117,231,198 | 388.260 | PASS |
| 13 | 1191,287,2137 | 579.120 | PASS |
| 14 | 134,1314,2354 | 766.760 | PASS |
| 15 | 8279,3509,2196 | 29524.340 | PASS |

脚本按现有 TBest 输出本地**估算**均分 `16.345`、平均吞吐 `537.67 GOPS`。
当前本地工作量不单调，TBest 源于未知线上尺寸，因此该均分不代表线上得分，
也未达到约 30 分目标。

### 修改前后 msprof

在 `build/` 内，对应尺寸的输入文件由测试脚本生成：

```bash
msprof --output=profile_before_case8 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 78 155 70
msprof --output=profile_after_case8 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 78 155 70
msprof --output=profile_before_case15 --aic-metrics=Memory ./quant_matmul_relu_quant_custom 8279 3509 2196
msprof --output=profile_after_case15 --aic-metrics=Memory ./quant_matmul_relu_quant_custom 8279 3509 2196
```

- 第 8 点代表 task duration：342.387 -> 238.945 us；AIC MTE2 17.242 -> 13.115 us，
  AIV Vector 24.988 -> 18.150 us，但 AIV scalar 159.302 -> 191.944 us，
  MTE3 0.501 -> 9.616 us。说明少算一次 GEMM/少同步已降低总时长，
  但小尺寸瓶颈转移到 UB 中转保存及输出整理；不能把当前流水称为 Cube/AIV 已充分重叠。
- 第 15 点 task duration 34838.117 -> 29387.807 us；
  AIC 主存读 1.005 -> 1.367 GB/s，AIV UB 读 1.756 -> 2.642 GB/s。
  这些是 profiler 的带宽字段，不是 MTE2/MTE3 峰值利用率。
- 原始 profile 在忽略的 `build/profile_before_case*` 与 `build/profile_after_case*`。
  下一轮先针对宽 N 的二次 GEMM、AIV 标量保存、输出 MTE3 和双缓冲做有对照实验。

### 复现

```bash
cmake --build build -j4
cd build
python3 ../scripts/test_dimensions.py
python3 ../scripts/test_dimensions.py --patterns
python3 ../scripts/test_dimensions.py --boundaries
```

不要提交 `build/` 或生成的输入输出。保留用户已有 `main.asc`、测试脚本及旧笔记的其他改动。

### 2026-09-26 架构重构前实测（历史基线）

状态：完成硬件确认、当前源码重编译、15 点基线、msprof 采集和无调用旧 kernel 清理。
统一主 kernel 尚未实施：等待确认是否允许独立设备 workspace。旧说明禁止设备内存申请，
而本次要求同时覆盖任意 N、不复算 GEMM、不借用 y；在当前硬件上不能同时满足这些约束。
具体设计见 [UNIFIED_KERNEL_DESIGN.md](UNIFIED_KERNEL_DESIGN.md)。

### 硬件与环境

- `npu-smi info`：物理 NPU 7，双芯片，每芯片 HBM 65536 MiB。
- `aclrtGetSocName()`：`Ascend910_9362`。不是仅根据 npu-smi 的 Ascend910 名称推断型号。
- `aclrtGetDeviceInfo(0, 101/102/201, ...)`：20 AI Core、20 Cube、40 Vector。
- 对应 CANN platform_config：L0A/L0B 各 64 KiB，L0C 128 KiB，L1 512 KiB，UB 192 KiB，L2 168 MiB。
- platform_config 的 memory_size 为 32 GiB，实机 npu-smi 为 64 GiB；容量以实机查询为准。
- CANN `/home/developer/Ascend/cann-9.0.0`，asc-devkit Version=9.0.0。
- `build/CMakeFiles/quant_matmul_relu_quant_custom.dir/flags.make`：`--npu-arch=dav-2201`。

### 当前 15 点基线

从用户提供的工作区源码重新 `cmake --build build -j4`，保持原始输入生成器和精度标准。
`python3 ../scripts/test_dimensions.py` 在第 8 点失败退出；其余点逐个调用同一脚本继续收集。

| 点 | M,N,K | ACL event 时间 (us) | 精度 |
|---:|---|---:|---|
| 1 | 59,35,56 | 574.040 | PASS |
| 2 | 46,43,62 | 618.640 | PASS |
| 3 | 67,64,33 | 202.360 | PASS |
| 4 | 107,58,70 | 429.020 | PASS |
| 5 | 79,55,105 | 1253.840 | PASS |
| 6 | 76,96,89 | 75.620 | PASS |
| 7 | 67,79,153 | 2261.780 | PASS |
| 8 | 78,155,70 | 343.900 | FAIL, 2159 个 y 差异 |
| 9 | 61,167,97 | 372.420 | FAIL, 887 个 y 差异 |
| 10 | 124,166,127 | 249.080 | FAIL, 1376 个 y 差异 |
| 11 | 138,269,95 | 511.900 | FAIL, 2133 个 y 差异 |
| 12 | 117,231,198 | 642.720 | FAIL, 1291 个 y 差异 |
| 13 | 1191,287,2137 | 1144.860 | FAIL, 22810 个 y 差异 |
| 14 | 134,1314,2354 | 1218.420 | FAIL, 14776 个 y 差异 |
| 15 | 8279,3509,2196 | 34321.219 | FAIL, 626730 个 y 差异 |

结论：7/15 PASS，不满足计分条件，不报告有效平均分。脚本使用来自未知线上尺寸的 TBest，
不能把当前本地随机维度的估分解释为线上得分。第 13/14/15 点实际工作量分别为
730462929、414482904、63796020156，脚本旁边的工作量注释过时，且实际工作量不严格递增。

### Profiling

在 `build/` 内、输入文件与命令尺寸匹配时执行：

```bash
msprof --output=profile_before_case8 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 78 155 70
msprof --output=profile_before_case15 --aic-metrics=Memory ./quant_matmul_relu_quant_custom 8279 3509 2196
```

- 第 8 点最后一个 task：336.047 us；AIC MMAD 1.045 us，AIC MTE2 14.766 us；
  AIV Vector 24.988 us，标量 159.100 us，MTE2 4.595 us，MTE3 0.441 us。
  AIC MAC 占比 0.003，AIV Vector 占比 0.075，标量占比 0.477；
  profiler 另列的 cube_utilization 为 9.816%，与 MAC 时间占比不是同一指标。
- 第 15 点 Memory 采样中，一个预热后 task 为 34308.785 us；
  AIC 主存读/写 1.020/0.026 GB/s，AIV 主存读/写 0.011/0.005 GB/s。
  这些是 profiler 输出的主存带宽字段，不能直接称为 MTE2/MTE3 峰值带宽利用率。
- 原始数据保存在 `build/profile_before_case*`，不纳入源码提交。
- 尚无统一架构的修改后 profiling 或提速结论。

### 已完成的独立清理

删除 `quant_matmul_relu_quant_rowmax_2d`、`quant_matmul_relu_quant_cube128`、
`quant_matmul_relu_rowmax_mix`、`quant_matmul_relu_quant_parallel_mix`、
`quant_expand_packed_output`、`quant_matmul_relu_quant_tail`。它们均没有当前 host 调用者。
`kernel.asc` 从 1993 行降到 943 行；保留当前两个实际使用的 kernel，未声称修复其已知错误。
保留用户原有 main.asc、测试脚本及此前笔记改动。
清理后 `cmake --build build -j4` 和 `git diff --check` 通过。
抽查 `python3 ../scripts/test_dimensions.py 1 6 8`：第 1/6 点通过，575.020/74.580 us；
第 8 点 343.480 us，仍为完全相同的 2159 个 y 差异和 12 个 scale 差异。
该清理不改变主路径算法，尚未修复基线故障；没有宣称完整回归通过。

### 当前版本（历史快照）

- `run_kernel` 保持评测要求的固定 14 参数接口。
- 每次 `run_kernel` 调用只发射一个设备 kernel，满足 profiling 的单次发射规则。
- 本轮实机：Ascend 910（NPU ID 7，双芯片物理设备，每芯片 20 个 AI Core），HBM 64 GB/芯片；CANN `/home/developer/Ascend/cann-9.0.0`，编译架构 `dav-2201`。
- `N >= 128` 且 `K <= 12288` 时优先使用 Cube 路径；其余形状使用 AIV 标量回退。
- 大矩阵主路径是 `KERNEL_TYPE_MIX_AIC_1_1`：一个 AIC 完成 INT8 Cube，配套 AIV 完成反量化、ReLU、动态 scale 和 INT8 输出。
- 输出区借用作 INT32 scratch，因此每个 tile 都检查 scratch 的跨行边界；结果写回后再恢复被覆盖的 INT8 区域。
- 15 个标定用例和 6 个输入模式专项均通过 NumPy golden。

## 计算和调度

Cube 使用 `M_TILE=16`、`K_TILE=256`，N tile 根据形状选择 64、128 或 192。大 K 形状用 `MTE2 -> MTE1 -> M` 与 `M -> MTE1` 事件重叠下一块 B 的搬运和当前 MMAD；B1 在下一次覆盖前用 `MTE1 -> MTE2` 保护。192 列是当前设备上稳定的上限，增大到 256 会触发 `507015`。

`K` 尾块向上补齐到 32，ND2NZ 的补零保证 MMAD 读取安全。`N` 尾列向左移动到完整 16 列 fractal，最后一个尾块最多多算 15 列；一个已复现的 32 列尾块边界会额外扩成 48 列，覆盖 `N=529..543` 等第二个完整 M tile 的尾部错误。

`M % 16 == 1` 的最后一行不走不稳定的 Cube 小 M 尾块，而由同一个 kernel 内的 AIV 分片完成点积、行最大值和量化。其他 M 尾数仍让 MMAD 使用完整 16 行，Fixpipe 只写真实行数。

## 输出和数值

非 32 字节对齐的 INT8 输出不能直接使用 MTE3 写任意行首。`StorePackedBytes` 先处理首尾字节，中间按 64 位 GM 写入，并对 UB 源地址非 8 字节对齐的片段用两个 64 位字拼接。连续主体展开为 64 字节和 32 字节两段，避免逐字节 `GetValue`。

对于 `N % 16 == 0` 的写回，只有目标地址、UB 源地址和长度同时满足 32 字节条件时才使用 `DataCopyPad`；否则仍使用打包标量写，避免 MTE3 覆盖相邻行。

AIV 的量化先走向量 `CAST_RINT`，再把半整数候选掩码按 64 位扫描。只有非零掩码块进入标量 round-half-to-even 修正，窗口约为 `1e-5`；窗口放宽到 `1e-4` 会在隐藏数据上产生少量错误。

## 本地标定结果

### 测试配置（2026-09-26 更新）

测试脚本：`scripts/test_dimensions.py`。从 `build/` 运行：

```bash
python3 ../scripts/test_dimensions.py
python3 ../scripts/test_dimensions.py --patterns
python3 ../scripts/test_dimensions.py --regenerate-cases <seed>  # 重新生成随机维度
```

**测试集设计原则**：
- 工作量基于线上实际运行时间按比例计算，确保本地得分与线上得分（~11.82）接近
- 15 个测试点工作量严格单调递增（17,442 → 6,662,656）
- 每个测试点的 M、N、K 维度随机分配，避免过拟合特定矩阵形状
- 覆盖小矩阵、大矩阵、非对齐维度、不同 M/N/K 比例等多种情况

**线上参考数据**（2026-09-26 提交）：
- 平均分：11.82
- TBest 时间范围：2.28μs - 542.29μs
- 实际运行时间：53.06μs - 20.4ms

最近一次完整运行结果（本地性能基线）：

| 测试点 | M,N,K | 工作量 | Kernel 时间 | 吞吐量 |
|---:|---|---:|---:|---:|
| 1 | 34,27,19 | 17,442 | 141.10 μs | 0.25 GOPS |
| 2 | 20,38,23 | 17,480 | 133.66 μs | 0.26 GOPS |
| 3 | 16,28,40 | 17,920 | 106.88 μs | 0.34 GOPS |
| 4 | 32,33,17 | 17,952 | 146.90 μs | 0.24 GOPS |
| 5 | 48,32,50 | 76,800 | 143.88 μs | 1.07 GOPS |
| 6 | 55,41,35 | 78,925 | 461.10 μs | 0.34 GOPS |
| 7 | 64,48,50 | 153,600 | 210.34 μs | 1.46 GOPS |
| 8 | 69,105,81 | 586,845 | 1838.42 μs | 0.64 GOPS |
| 9 | 80,80,92 | 588,800 | 414.30 μs | 2.84 GOPS |
| 10 | 64,91,102 | 594,048 | 2132.94 μs | 0.56 GOPS |
| 11 | 174,224,117 | 4,560,192 | 2055.92 μs | 4.44 GOPS |
| 12 | 224,224,91 | 4,566,016 | 1419.68 μs | 6.43 GOPS |
| 13 | 208,128,173 | 4,605,952 | 1472.76 μs | 6.25 GOPS |
| 14 | 221,204,103 | 4,643,652 | 1539.72 μs | 6.03 GOPS |
| 15 | 208,224,143 | 6,662,656 | 2189.28 μs | 6.09 GOPS |

**当前本地性能**：
- 估算平均分：**12.47**（接近线上 11.82）
- 平均吞吐量：2.48 GOPS
- 工作量单调性：✓ 通过
- 精度验证：✓ 所有 15 个测试通过

**性能瓶颈观察**：
- 小矩阵（测试点 1-7）：吞吐量 0.24-1.46 GOPS，启动开销占比高
- 中等矩阵（测试点 8-10）：性能波动大（0.56-2.84 GOPS），调度策略不稳定
- 大矩阵（测试点 11-15）：吞吐量 4.44-6.43 GOPS，仍远低于理论峰值
| 10 | 288,2432,9984 | 1023.84 us |
| 11 | 353,512,2201 | 1208.74 us |
| 12 | 320,4096,3841 | 1659.94 us |
| 13 | 28160,2049,32 | 34594.56 us |
| 14 | 640,6400,7008 | 7349.86 us |
| 15 | 640,9216,11840 | 15573.92 us |

按题目公式和最新 TBest 计算，本地平均估分为 `13.485`。最新线上基线为 `10.88` 分，15 点时间为 `52.87, 9.12, 45.72, 38.77, 15000, 139.93, 497.81, 7720, 439.67, 1120, 61600, 1572000, 62800, 7420, 18100 us`。本地用例按这些时间的量级重新构造，仍会受到线上隐藏维度和设备负载影响。

### 已验证的优化（历史快照）

1. 从 Cube 两遍矩阵乘保留单 kernel，避免 profiling 统计到额外 launch。
2. 用 64 位读取、拼接和展开写回替代非对齐 N 的逐字节输出，点 13 从约 42 ms 降到约 34.6 ms。
3. 用 `MTE2 -> MTE1 -> M`、`M -> MTE1` 和 `MTE1 -> MTE2` 事件重叠大 K 的 B 搬运，点 14 从约 7.95 ms 降到约 7.35 ms，点 15 从约 16.85 ms 降到约 15.57 ms。
4. 合并向量 Muls、Mul、ReLU 的中间屏障，并按 64 位扫描半整数候选掩码。
5. 使用 `msprof --output=msprof_case14b ./quant_matmul_relu_quant_custom 640 6400 7008` 和对应的 case 12 命令采集指标：Cube 利用率约 99.9%，MTE2 是主要耗时项（约 62%~66%），因此将 `n >= 512` 的 Mix 路径 tile 从 128 调整为 192。case 6 约从 224 us 降至 212 us，case 12 稳定在约 1.59 ms；case 5、11、14 无明显回归。

### 本轮回归（历史快照）

- 修改：`kernel.asc` 中 `n >= 512` 的优先 `tileWidth` 使用 192；`m >= 128 && k >= 3000` 也使用 192。
- 15 点全部 PASS，最近一次完整运行平均估分约 `12.46`；6 个 pattern 专项全部 PASS。
- 当前仍低于约 30 分目标，主要剩余瓶颈是 Mix 路径的 MTE2 搬运和 `M % 16 == 1` 尾行的 AIV 标量点积；继续优化前应保持 msprof 对比并覆盖非对齐尺寸。

### 提交边界（历史快照）

提交时只保留源码和脚本，不提交 `build/`、`scripts/__pycache__/` 或本地生成的 `input/`、`output/`。不要在 `kernel.asc` 中增加 `main()`、设备内存申请、workspace 参数或第二次 kernel launch。

### 2026-09-26: 本地测试工作量调整

### 目标
重新设计本地测试，使其工作量梯度与线上测试一致，同时保持维度随机性。

### 策略
1. **工作量基准**：基于线上 TBest 时间（2.28μs - 542.29μs）按比例计算工作量
2. **单调递增**：强制工作量从测试点 1 到 15 严格递增（7130 → 1843200）
3. **维度随机化**：在保持总工作量不变的前提下，随机分配 M、N、K
4. **不强求时间匹配**：接受本地运行时间与线上有差异（kernel 还在优化中）

### 测试结果
- 所有 15 个测试通过精度验证
- 工作量范围：7,130 → 1,843,200（严格递增 ✓）
- 当前平均分：22.29（基线）
- 平均吞吐：1.46 GOPS

### 后续优化方向
- 小矩阵（测试点 1-9）性能较低，需要减少 kernel 启动开销
- 大矩阵（测试点 10-15）存在严重瓶颈，需要 profiling 分析
- 目标：将平均分从 22.29 提升到 30+


### 2026-09-26 16:30 - Optimization Round 3: Re-enable Cube + Mix kernel

### Changes
1. Re-enabled Cube path by setting `cubeEligible = n >= 128 && k <= 12288`
2. Lowered Mix kernel threshold from N >= 256 to N >= 128
3. Removed buggy Cube128 fallback, replaced with AIV fallback

### Kernel dispatch strategy
- **Small N (< 128)**: AIV-only (`quant_matmul_relu_quant_custom`)
- **Medium/Large N (>= 128), Small/Medium K (<= 12288)**: Mix kernel (Cube + AIV fusion)
- **Large K (> 12288)**: AIV-only (Cube inefficient for large K)

### Performance results (local test)
```
Test  Dimensions      Workload   Time(μs)  Throughput  vs Previous
--------------------------------------------------------------------
01    (34,27,19)      17,442     141.46    0.25 GOPS   ~same
02    (20,38,23)      17,480     133.74    0.26 GOPS   ~same
03    (16,28,40)      17,920     106.74    0.34 GOPS   ~same
04    (32,33,17)      17,952     147.22    0.24 GOPS   ~same
05    (48,32,50)      76,800     143.68    1.07 GOPS   ~same
06    (55,41,35)      78,925     460.40    0.34 GOPS   ~same
07    (64,48,50)      153,600    209.36    1.47 GOPS   ~same
08    (69,105,81)     586,845    1841.20   0.64 GOPS   ~same
09    (80,80,92)      588,800    413.98    2.84 GOPS   ~same
10    (64,91,102)     594,048    2136.72   0.56 GOPS   ~same (AIV)
11    (174,224,117)   4,560,192  116.32    78.41 GOPS  18.1x faster!
12    (224,224,91)    4,566,016  118.30    77.19 GOPS  new (Mix)
13    (208,128,173)   4,605,952  49.00     188.00 GOPS 49.0x faster!
14    (221,204,103)   4,643,652  120.54    77.05 GOPS  new (Mix)
15    (208,224,143)   6,662,656  115.02    115.85 GOPS 6.2x faster!

✓ All 15 tests PASSED
Average score: 3.070 (was 22.29, 7.3x improvement)
Average throughput: 36.30 GOPS (was 1.46 GOPS)
```

### Root cause analysis
**Previous issue**: Disabled all Cube kernels due to Cube128 N-tail bugs
**Solution**: Use Mix kernel (flexible tileWidth) instead of Cube128 (fixed N_TILE=128)

Mix kernel advantages:
- Adjustable tileWidth (16-192) adapts to different N
- Better tail handling for non-aligned N
- Cube does GEMM, AIV does dequant/ReLU/quant in fusion
- Reduces GM traffic by keeping intermediate FP32 in UB

### Why large matrices improved dramatically
1. **Cube GEMM**: 10-50x faster than AIV-emulated matrix multiply for large tiles
2. **Fusion**: No intermediate GM write of INT32/FP32 matrices
3. **Better parallelism**: Mix kernel splits work across AIC+AIV cores effectively

### Next optimization opportunities
1. **Test 10** still slow (2.1ms for 594K workload): N=91 < 128 forces AIV-only
   - Could lower Cube threshold to N >= 64 if Mix kernel supports smaller tiles
2. **Small tests (1-6)** unchanged: overhead-dominated, need different optimization
3. **Tile size tuning**: Current preferredTileWidth heuristic may not be optimal
4. **K-axis tiling**: Large K could benefit from K-tile optimization


### 优化轮次 7: 降低 Cube 使用阈值（N >= 80）

### 修改
- 将 `cubeEligible` 从 `n >= 128` 降低到 `n >= 80`
- 保持 `tileWidth >= 16` 的安全检查
- 更多中等尺寸矩阵可以使用 Mix kernel（Cube + AIV）

### 性能结果
```
Test  Before(μs)  After(μs)  Speedup  Notes
09    414.26      70.12      5.9x     N=80 now uses Cube
10    2129.28     300.60     7.1x     N=91 benefits from Cube
11-15 113-120     113-118    1.0x     Large matrices unchanged
```

**平均分：3.08 → 4.67（+52% 提升）**

### 分析
1. **中等矩阵获益显著**：N ∈ [80, 128) 范围现在使用 Cube 硬件
2. **大矩阵保持稳定**：N >= 128 的性能未退化
3. **小矩阵仍需优化**：测试点 1-8 仍然较慢（< 2 GOPS）

### 瓶颈识别
- 测试点 1-7：N < 80，仍走纯 AIV，延迟 100-460μs
- 测试点 8：N=105 但 tileWidth 可能受限，仍较慢（1831μs）
### 实验：N_TILE 128 -> 192（失败，已回退）

理论：第 15 点 N=3509 时 128 列约 28 个 tile，192 列约 19 个 tile，
可减少 B 的 MTE2/MTE1 搬运；实测 profile 显示第 15 点 MTE2 9.27 ms、MAC 0.66 ms。
修改 `N_TILE=192` 后，专项 `64x7808x1024` 在同步时触发设备错误 **507015**，
尚未进入完整性能测试；恢复 `N_TILE=128`，避免不稳定 L0/事件资源配置。

### 实验：按 scratch 容量扩大 rowGroup（保留）

理论：原 host 逻辑固定以 `groupRows*n/(16*4)` 选 tileWidth，导致 N=128 时
强制 32 列、4 个 tile；这不是 L0 限制，而是为了让 16x32xINT32 scratch 不越过
16 行组。新调度先选完整 128 列，再把 rowGroup 增到满足
`rowGroup*n >= 16*tileWidth*4`：N=128 使用 64 行组，N=256 使用 32 行组，
scratch 恰好覆盖整个 task 的输出前缀。主 kernel 显式接收 rowGroup，减少 tile 和
CrossCore 交接，不增加 workspace。先覆盖 M 小于/大于 rowGroup、N=128/256、
奇数 N 和多 block 负载。

实现后的专项：`1024x128x800` 77.56--78.52 us，`1024x256x257` 109.10--114.46 us；
`128x16/32/48/64/17x80` 全部 PASS。`msprof` 的 `1024x128x800` task 约 90.50 us、
16 blocks，`1024x256x257` task 112.18 us、20 blocks，证明大 rowGroup 没有减少并行
block 数到单核。完整校准 15 点仍 15/15 PASS，单次均分 12.904--12.977，
与 12.953 基线在设备测量噪声范围内；收益主要针对线上未公开的 N=16/32/64/128/256。

修正记录：第一次实现用 `ceil(N/16)`，在 `M>=128,N=55` 产生 `tileWidth=64>N`
和 507015；现改为 `floor(N/16)*16`，并在扩 rowGroup 时继续保持
`(rowGroup*N)%64==0`。`M=64/128,N=55/63/64` 及完整 22 个 boundaries 全部 PASS。

资源探测：`N_TILE=160` 和 `N_TILE=144` 在 `64x7808x1024` 均触发 507015，
已回退到 128；这确认当前组合的稳定 L0 边界，后续不再尝试单纯放宽 N tile。

最终还原验证：`M=100,N=128,K=12289`、`17x129x12289`、`2x17x12289` 全部 PASS；
fallback 的 block 计算使用原始对齐 rowGroup，避免 Mix 的扩展 rowGroup 降低超大 K
回退路径并行度。当前源码已重新编译，`git diff --check` 无错误。

### 新线上结果校准（2026-09-26 20:19）

线上最新耗时为 `40.40, 7.56, 35.08, 35.21, 85.48, 122.90, 417.36,
490.52, 461.43, 1090, 3940, 4160, 3360, 7100, 14200 us`，按当前评分公式
估算线上分数约 **13.307**。已更新 `scripts/test_dimensions.py` 的 ONLINE_ACTUAL_US
和 CASES，保持宽 N、奇数 N、短 K、大 K 等覆盖：第 7/8/10/13/14/15 点本地分别
约 413/497/1089/3233/6663/14151 us；第 2 点受最小 kernel 启动下限影响，
本地约 14--24 us，不能由正整数维度稳定复制线上 7.56 us。
第 11 点 `K=6200` 测得 5442 us，改用 `K=4500`；单独测 `K=5000` 为 4406 us、
`K=5500` 为 4884 us，预计新维度更接近线上 3940 us。
校准后完整 15 点实测：15/15 PASS，估算本地均分 **12.953**；第 1--15 点耗时
约 `38.60,14.50,33.64,33.94,80.48,128.94,415.34,498.56,491.04,
1084.18,4008.02,3747.62,3232.84,6791.10,14155.50 us`。第 2 点为启动下限，
其余宽 N/大 K 点已与新线上量级对齐。

### 实验：K_TILE 384 -> 416（失败，已回退）

理论：416=13x32，B2 单块约 53 KiB，仍低于 C220 资源边界；K=7008 时循环
预计由 19 次降到 17 次，减少 MTE2/MTE1 握手。优先测试大 K 及非对齐 K，
若出现 507015、精度差异或均分回退则恢复 384。
实际第 11 点 `576x5760x4500` 在同步时触发 **507015**，未进入完整性能测试；
恢复 `K_TILE=384`。这表明当前 A1/B1/C1 组合下 416 不是稳定可用边界。
### 实验：row group 内 B1 生命周期复用（失败，已回退）

大 K profile 显示 MTE2 是主瓶颈，但直接增大 N/K tile 会触发 507015。对短/中 K
采用不同的生命周期：当 `K<=384` 且一个 128 列 tile 的 NZ B 不超过 160 KiB 时，
在 row group 开始预装所有 K tile 到 B1，多个 M tile 复用；原有 `K<=64` 的全 N
预装路径不变。该路径只增加 L1 生命周期，不申请 workspace，需验证 B1 多 K tile
偏移、尾 K、奇数 N 和跨 row tile 的精度。
实际实现按 `kBase*tileWidth*alignK` 叠放多个 NZ B tile，最小 `55x55x105` 即
出现 1470 个 `y` 差异、44 个 scale 差异；说明 NZ 目标矩阵的实际 stride 不能
按该线性偏移推断。立即回退，未进入性能结论。下一次若重做必须依据 `Nd2Nz`
矩阵布局的真实 `dstNzMatrixStride` 或采用显式中间布局，不能继续堆指针偏移。

回退后的 smoke：`55x55x105`、`576x5760x4500`、`6784x3509x2196` 均 PASS，
耗时约 80.60/3987.74/14200.04 us。当前稳定参数为 `N_TILE=128`、`K_TILE=384`；
校准 CASES 的完整 15 点基线为 15/15 PASS、均分 **12.953**。

### 实验：兜底路径与分组生命周期（2026-09-27）

线上无明显收益的主要原因之一是小 `N`/小输出进入 `quant_matmul_relu_quant_custom`，
该路径无法安全使用 `y` 前缀保存 INT32 Cube tile，只能保留 AIV 标量点积。尝试过：

- 将标量累加从 `int64_t` 改为 8 路 `int32_t`。支持的 `K<=65537` 下乘积和仍在
  INT32 范围内；全部 patterns/boundaries 通过，小 case 约 1--3% 改善，保留。
- 将 A 行复制到 UB/本地数组以复用 GM 读取。实测 case1/3/4 退化到约
  46.8/39.2/39.6 us，原因是标量 `GetValue` 和额外 MTE2 开销，已回退。
- 试验 AIC-only 小 16 列 Cube，并直接标量读取 C1。C220 的 C1 不能按普通
  `GetValue` 读取，产生错误，已完全删除，不进入 host 调度。
- 尝试把 A1 从整行改为每个 K tile 流式装载，以覆盖 `K>12288`。`K=800` 等
  多 tile 形状出现错误，已回退；当前 `K<=12288` 条件仍是经过验证的 Cube 上限。

为改善中小矩阵多核利用，随后将 `rowGroup*n` 的保守 64B 对齐放宽到 32B，且新增
`GroupCount`：默认使用 ceil 分组，只有最后一组不足一个 INT32 scratch tile 时才
退回 floor 分组。这样 `55x55` 理论上可拆成 `32+23`，而 `65x2049` 仍保持安全的
单组 scratch 生命周期。`55x55x105`、`65x2049x257`、`65x543x257`、奇数尾块
专项均 PASS；完整 patterns/boundaries 需在提交前再次运行。

本轮硬件确认：`npu-smi 25.5.5` 报告 Ascend910、HBM 64GB；编译目标仍为
`--npu-arch=dav-2201`，CANN 为 `/home/developer/Ascend/cann-9.0.0`。再次运行
`msprof --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 6784 3509 2196`
时应用侧报 `Operation not permitted`，虽生成 profiling 目录但没有 summary 数据，
因此本轮不虚构 Cube/Vector/MTE 百分比；可用依据仍是 ACL event kernel duration
和此前成功采集的 MTE2 主瓶颈结论。

复核完成：本轮完整 15 点、12 个 patterns、22 个 boundaries 均 PASS；15 点一次测得
约 `38.62,14.60,33.94,33.24,80.50,120.20,421.86,491.96,443.98,614.30,
2262.44,2152.74,2944.92,3528.12,7047.94 us`，脚本估算均分 **13.729**。

### 2026-09-27 中小尺寸通用调度优化

硬件和工具链未变化：Ascend910_9362，20 Cube/40 Vector，UB 192 KiB，L1 512 KiB，
CANN 9.0.0，`dav-2201`。本轮先清理了已删除但仍残留的 `SmallPaddedKernel` 调度调用，
随后只保留两类可验证改动：

1. 对 `M<=16,N<=16,K<=128`、工作量 `M*N*K<=12288` 且没有现成 `SmallKernel` 对齐
   特化的形状，复用已验证的 `HugeKKernel` AIV 路径。该路径覆盖 `K=16/80` 等小而非
   32B 对齐的输入，避免 Cube 固定 tiling/accumulator 交接；已有
   `K=32/64/128,N=16/32/64/128` 路径不被抢占。
2. 对未进入小 Vector 特化、且 `M<=64,N<=128,K<=256` 的 Cube 路径，将 `tileM` 从按核
   数得到的碎片值提升到最多 8。这样减少极小行块的 Matmul/Fix/Vector 交接次数，同时不
   改变 accumulator 布局或量化顺序。

曾尝试将任意 `K=16/80` 直接作为 `SmallKernel` 编译期参数，但 `DataCopy` 对非 32B
对齐长度的语义不满足该 kernel 前提；`1x16x80` 出现 y/scale 错误，已完整回退。
也曾让 `HugeKKernel` 处理 `55x55x105`，实测超时，已回退，仅保留极小工作量门控。

专项结果（`cd build && python3 ../scripts/test_dimensions.py --shape ...`）：

| Shape | Kernel time (us) | Result |
|---|---:|---|
| `1x16x80` | 约 14.3 | PASS |
| `3x8x16` | 约 14.3 | PASS |
| `16x16x16` | 约 39.3 | PASS |
| `8x16x64` | 约 14.7 | PASS，保留原 SmallKernel |
| `55x55x105` | 约 47.9 | PASS |
| `17x31x33` | 约 45.3 | PASS |

完整回归：15/15 主测试 PASS，12/12 patterns PASS，22/22 boundaries PASS。最终一次
15 点耗时（us）为 `42.28,14.34,39.12,14.48,55.94,104.28,72.16,69.08,123.82,
479.56,518.04,1003.22,758.00,1028.74,1662.76`，脚本估算均分 **19.885**；设备
负载波动明显，专项重复测量仍确认小 AIV 和 `tileM=8` 方向的收益。

Profiling 命令：

`msprof --output=/tmp/qmrq_profile_case5 --aic-metrics=PipeUtilization ./quant_matmul_relu_quant_custom 55 55 105`

ACL event 得到一次 `64.34 us`（普通测试一次为 `50.18 us`）；本环境成功采集并解析
设备目录，但没有导出 `summary`，因此不报告虚构的 Cube/Vector/MTE 百分比。代码检查
使用 `git diff --check`，无空白错误；build/profile 目录不纳入提交。

### 2026-09-27 形状无关的中小尺寸与宽矩阵修复

本轮仍在 Ascend910_9362（20 Cube/40 Vector，UB 192 KiB，L1 512 KiB，CANN 9.0.0，
`dav-2201`）上验证。线上真实形状不可见，因此只保留不依赖具体尺寸的调度规则。

1. **通用 Cube 行块下限**：原 `tileM=(M+cores-1)/cores` 会产生小于 8 的行块。
   对非 16 对齐的 M，这会使 Cube/AIV 累加槽和向量行分工出现不稳定；`tileM<8` 时
   统一提升到 8。`134x1314x2354` 曾稳定出现全行错误，提升后恢复 bit-exact，
   并通过 `134x1314x2354`、`174x224x117`、`161/199/300x269x95` 专项。

2. **Epilogue Vector 子核均分**：原批处理从 `start*batch` 开始，`rows=8,batch=8`
   时 subcore 0 处理 8 行、subcore 1 处理 0 行。现在先按 `rows*sub/2` 划分连续行区间，
   再在区间内批处理。`138x269x95` 从约 77 us 降到约 54--55 us，且 y/yScale 对拍通过。

3. **宽矩阵 K padding 门槛**：`N=8192..16383` 的小 K 非 32 对齐输入不再启动额外
   `PadInputsKernel`；`N>=16384` 仍保留 padding（`64x16385x105` 约 93 us，关闭
   padding 的一次测量约 108 us）。临界形状 `18x8193x33`、`64x8193x65`、
   `64x16385x105`、`33x16401x65` 均 PASS。

本轮验证：15/15 主 CASES、12/12 patterns、22/22 boundaries 全部 PASS；旧记录中的
15 个候选线上形状（包括 `78x155x70`、`67x64x33`、`138x269x95`、`134x1314x2354`）
全部 PASS。最新校准 CASES 一次耗时约 `42.40,14.72,39.50,14.52,52.46,88.36,
72.96,68.80,130.52,485.40,522.30,1102.44,773.30,1040.02,1658.08 us`，设备噪声下
估算均分 19.876；该分数不等同于线上隐藏形状得分。

曾实验运行时 N/K 的通用 Vector 面板，但 C220 对动态/非对齐 GM 行搬运存在 507035 或
结果差异，已禁用且不进入实际调度。不要将该实验路径重新打开，除非先用逐行输入 guard
和完整 golden 对拍重新证明正确性。

### 2026-09-27 短 K、N=1 与非对齐小矩阵覆盖

本轮在 Ascend910_9362（20 Cube/40 Vector，UB 192 KiB，L1 512 KiB，CANN 9.0.0，
`dav-2201`）上重新编译，并用以下命令采集设备指标：

```bash
cmake --build build -j4
cd build
msprof --output=/tmp/qmrq_profile_probe_case8new --aic-metrics=PipeUtilization \
  ./quant_matmul_relu_quant_custom 78 155 70
msprof --output=/tmp/qmrq_profile_hugek_current --aic-metrics=PipeUtilization \
  ./quant_matmul_relu_quant_custom 1 3 135169
```

`78x155x70` 的当前路径为 `WideKernel`，6 个 MIX_AIC 任务，单任务约 15--25 us；
`aicore` 的 Cube 利用率约 10--15%，AIV Scalar 约 4--7 us，说明把两个基础列块
强行合并不会稳定获益。强制普通行块的 A/B 实验由约 65 us 退化到约 64--74 us，
已回退。

保留的调度和边界修复：

1. `M<=128,N<=128,K<=16` 使用 `SmallKernel<K=32>` 的零填充 Vector 路径；对
   `N>16` 仅在 `M<=32` 时启用，避免中等矩阵出现 AIV Scalar 退化。
2. `65<=K<128` 的 Vector 覆盖扩展到 `M<=32` 的任意 `N<=128`；`K=128` 对非
   16/32/64 列宽使用 16/32/64/128 物理宽度并按真实 N 读写，避免尾块回退 Cube。
3. 所有 `N=1` 小矩阵改用 `N=8` 物理模板并以 `sourceN=1` 写回。原 `N=1` 模板在
   `1x1x17`、`16x1x32` 等形状的 yScale 会错误；修复后 `K=17/33/64/65/128`、
   `M=1/16/32` 专项全部 PASS。
4. `SmallBatch2Kernel` 保留真实 `sourceN` 参数和逐行尾块搬运代码，但入口仅对齐
   N 复用两行批处理；未验证的多行非对齐尾块不强行启用。

一次完整本地结果（每点 bit-exact，yScale 在容差内）为：

```text
18.62, 14.62, 14.96, 14.60, 55.82, 91.88, 72.60, 72.74,
130.74, 476.98, 492.54, 992.14, 747.46, 1025.12, 1610.90 us
```

估算均分 20.592。专项 `N=1`、`49x17x33`、`49x31x33`、`64x127x65`，以及
`python3 ../scripts/test_dimensions.py --patterns`（12/12）和
`python3 ../scripts/test_dimensions.py --boundaries`（22/22）全部通过。

曾尝试对 `K>65535,N<=512` 增加 128 项 Vector 分块点积；`1x3x135169` 的
`HugeKKernel` profile 显示 AIV Scalar 99.9%，但原型运行触发 507035，已完全从
入口移除，当前仍使用经过验证的 int64 标量兜底。该大 K 路径是后续优化重点，任何
重做必须先解决 C220 的动态 GM->UB 搬运限制并通过 overflow golden。
