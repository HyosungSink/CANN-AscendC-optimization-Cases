# 官方题面存档（三题全文）

> **来源**：用户在 2026-09-19 会话中直接提供的比赛平台题面全文。
> **性质**：**官方口径的权威原文**。本仓库其他文档与本文冲突时，**以本文为准**（除下文标注的"真机实测推翻项"）。
> 本文件取代此前丢失的 `OFFICIAL_PROBLEM_STATEMENT.md`（仅含第三题，已无备份）。


**相关文档**：[AGENT.MD](AGENT.MD)（入口） · 各题的状态与实测结论：[code1.md](code1.md) · [code2.md](code2.md) · [code3.md](code3.md) · [refs/README.md](refs/README.md)
> ⚠️ 本文是**官方口径**；凡与 `codeN.md` 的实测结论冲突处，以 `codeN.md` 为准（冲突清单见文末附录）。
---

# 第一题（B组简单题）mHC-expand 算子（前向与反向）

## 1. 算子说明

MHC Expand 是 Manifold HyperConnection（流形超连接）架构中的扩流算子，用于将残差连接的单一数据流扩展为多路并行流。在 DeepSeek 提出的 mHC（Manifold-Constrained Hyper-Connections）架构中，标准残差连接被扩展为多路并行残差流，以增强层间信息交互能力。本算子负责在前向传播时将输入张量从 `(S, D)` 扩展为 `(S, m, D)`，在反向传播时将梯度从 `(S, m, D)` 归约回 `(S, D)`。

该算子位于 mHC 架构的残差流扩展阶段：

```
Input x: [S, D]
        ↓
MHC Expand Forward（本算子前向）：将单流扩展为 m 路并行流
        ↓
o: [S, m, D] → 送入后续 H^res 混合矩阵计算
        ↓
MHC Expand Backward（本算子反向）：将 m 路梯度归约为单流梯度
        ↓
x_grad: [S, D]
```

参考资料：

- 代码实现：`https://github.com/deepseek-ai/TileKernels/blob/main/tile_kernels/mhc/expand_kernel.py`
- 论文：mHC: Manifold-Constrained Hyper-Connections（arXiv:2512.24880）

## 2. 输入输出说明

### 2.1 前向算子（MHC Expand Forward）

**输入规格**

| 参数名 | 类型 | 数据类型 | 维度(shape) | 说明 |
|---|---|---|---|---|
| x | 必选输入 | bfloat16、float16 | `[S, D]` | 输入张量，S 为 token 数，D 为隐藏层维度 |

**输出规格**

| 参数名 | 数据类型 | 维度(shape) | 说明 |
|---|---|---|---|
| o | bfloat16、float16 | `[S, m, D]` | 扩展后的输出张量，m 为扩展倍数 |

**形状约束**

- S：token 数量，取值为正整数
- D：隐藏层维度，取值为正整数
- m：扩展倍数（mhc_mult），取值为正整数
- 输出 `o` 的第 m 个副本与输入 `x` 完全相同：`o[i, m, j] = x[i, j]`

**计算公式**

```
o[i, m, j] = x[i, j]    对所有 m ∈ [0, mhc_mult)
```

即将输入 x 沿第 1 维复制 m 份，每个副本内容相同。

### 2.2 反向算子（MHC Expand Backward）

**输入规格**

| 参数名 | 类型 | 数据类型 | 维度(shape) | 说明 |
|---|---|---|---|---|
| o_grad | 必选输入 | bfloat16、float16 | `[S, m, D]` | 上游传来的梯度张量 |

**输出规格**

| 参数名 | 数据类型 | 维度(shape) | 说明 |
|---|---|---|---|
| x_grad | bfloat16、float16 | `[S, D]` | 归约后的梯度张量 |

**形状约束**

- `o_grad` 的形状必须与前向输出的形状一致
- `x_grad` 的形状必须与前向输入的形状一致

**计算公式**

```
x_grad[i, j] = Σ_{m=0}^{mhc_mult-1} o_grad[i, m, j]
```

即将 m 个副本的梯度沿第 1 维**求和**归约。

## 3. 算子逻辑说明

### 3.1 前向：数据扩展阶段

- 将输入 x 的形状从 `[S, D]` 读取到本地缓存
- 对 `mhc_mult` 个副本，逐个将 x 数据写入输出 o 的对应位置
- 每个副本 `o[:, m, :]` 的内容与输入 x 完全相同

### 3.2 反向：梯度归约阶段

- 初始化累加缓存为零
- 遍历 `mhc_mult` 个副本，将每个 `o_grad[:, m, :]` 累加到缓存中
- 将累加结果写入 x_grad

### 3.3 参考实现关键策略

参考 DeepSeek 的 TileLang 实现，采用了以下分块策略：

- **分块大小**：`blk_n = 32`（token 维度分块），`blk_h = 128`（隐藏维度分块）
- **并行策略**：按 `(ceildiv(S, blk_n), ceildiv(D, blk_h))` 启动 Kernel
- **前向**：将 x 的分块读入 fragment，逐副本写入输出
- **反向**：将 fragment 初始化为 0，逐副本累加梯度，最终写回

## 4. 任务要求

- **精度保障**：针对不同 Data type 和不同 shape 维度，设计算子逻辑，保证算子精度正确
- **性能优化**：充分发挥系统带宽能力，算子性能更优。前向需优化数据复制效率，反向需优化归约求和效率
- **切分最优**：探索输入 tensor 的切分方式，找到不同输入 shape 场景下的最优解
- **泛化功能**：必须实现算子泛化功能，满足各类合法输入场景的计算需求
- **前向反向完整性**：需同时实现前向和反向算子，反向算子的梯度计算必须与参考实现一致

## 5. 功能示例

```python
import torch

# 示例1：前向扩展（基础用法）
# 输入形状：x=[4, 8]，mhc_mult=2
x = torch.randn(4, 8, dtype=torch.bfloat16)
mhc_mult = 2

# 前向：将 [4, 8] 扩展为 [4, 2, 8]
o = x.unsqueeze(1).expand(-1, mhc_mult, -1).clone()

# 输出形状：o=[4, 2, 8]
# 结果：o[:, 0, :] = x, o[:, 1, :] = x（两个副本内容相同）
# 验证：torch.allclose(o[:, 0, :], x) -> True

# 示例2：反向归约
# 输入形状：o_grad=[4, 2, 8]
o_grad = torch.randn(4, 2, 8, dtype=torch.bfloat16)

# 反向：将 [4, 2, 8] 归约为 [4, 8]
x_grad = o_grad.sum(dim=1)

# 输出形状：x_grad=[4, 8]
# 结果：x_grad[i, j] = o_grad[i, 0, j] + o_grad[i, 1, j]

# 示例3：大模型典型规模
# 输入形状：x=[4096, 7168]，mhc_mult=4
x = torch.randn(4096, 7168, dtype=torch.bfloat16)
mhc_mult = 4

o = x.unsqueeze(1).expand(-1, mhc_mult, -1).clone()
o_grad = torch.randn(4096, 4, 7168, dtype=torch.bfloat16)
x_grad = o_grad.sum(dim=1)
# 等价于：x_grad = o_grad[:, 0, :] + o_grad[:, 1, :] + o_grad[:, 2, :] + o_grad[:, 3, :]
```

## 6. 测试用例覆盖范围

- **数据类型**：bfloat16、float16
- **维度场景**：小规模 `(S=64, D=256, m=2)`、中规模 `(S=1024, D=4096, m=4)`、大规模 `(S=8192, D=7168, m=8)`
- **扩展倍数**：`m=2`、`m=4`、`m=8`
- **前向验证**：输出每个副本与输入是否一致
- **反向验证**：梯度归约求和是否正确
- **边界场景**：`S=1`（单 token）、`D=1`（单维度）、非对齐维度
- **精度场景**：float16 累加精度、bfloat16 累加精度

---

# 第二题（B组中等题）mHC-Sinkhorn 算子

## 1. 算子说明

MhcSinkhorn 算子基于 Sinkhorn-Knopp 迭代算法，将一个任意非负方阵变换为双随机矩阵（即每行元素之和、每列元素之和均为 1 的方阵）。该算子常用于深度网络中稳定信号传播、缓解梯度消失/爆炸问题，其输出可作为后续网络层的输入。

算子接收输入矩阵 x（形状最后两维为 n×n 方阵），对其进行 `numIters` 次交替的"行归一化—列归一化"迭代，最终输出双随机矩阵 output。迭代过程中还可输出归一化中间结果 normOut 与求和中间结果 sumOut，用于后续反向梯度计算。

## 2. 输入输出说明

### 输入规格

| 输入 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| x | 张量 | `(B, S, n, n)` 或 `(T, n, n)` | **float32** | 输入张量，最后两维为待归一化的 n×n 方阵；前序维度（T 或 B、S）为批量维度，各方阵独立计算 |
| eps | 属性 | 标量 | float32 | 归一化防除零参数，建议值 `1e-6` |
| numIters | 属性 | 标量 | int64 | Sinkhorn 迭代次数，控制收敛过程，取值范围 **1~100**，建议值 20 |

### 输出规格

| 输出类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|
| output | 与 x 一致：`(B, S, n, n)` 或 `(T, n, n)` | float32 | Sinkhorn 变换最终结果（双随机矩阵，行和、列和均为 1） |
| normOut | 逻辑形状 `(2numIters, B, S, n, n)` 或 `(2numIters, T, n, n)`；物理存储 `size = 2numIters·n·n_align·B·S` 或 `2numIters·n·n_align·T` | float32 | 迭代过程中每步的归一化中间矩阵，用于反向，可选（不输出时传空指针） |
| sumOut | 逻辑形状 `(2numIters, B, S, n)` 或 `(2numIters, T, n)`；物理存储 `size = 2numIters·n_align·B·S` 或 `2numIters·n_align·T` | float32 | 迭代过程中每步的行/列求和中间结果，用于反向，可选（不输出时传空指针） |

### 形状约束

- 输入 x 仅支持 3 维 `(T, n, n)` 或 4 维 `(B, S, n, n)`，其他维度数不支持。
- 矩阵维度 n 仅支持取值 **4、6、8**（即输入最后两维的大小）。
- `numIters` 取值范围为 **1~100**，超出范围报参数无效错误。
- **仅支持 FLOAT32 数据类型**与 ND 数据格式（即任意多维非结构化连续格式），**不支持 float16/double 等其他精度**。
- normOut 与 sumOut 为可选输出，传空指针时不输出对应结果。
- 输入含 `-inf/inf/nan` 时，**对应位置输出 nan**。
- 算子默认采用**确定性实现**，相同输入多次调用结果一致。

### 输出形状计算公式

- output 形状与输入 x 完全一致。
- normOut 在 x 形状前增加一维 `2*numIters`：逻辑形状为 `(2*numIters, B, S, n, n)` 或 `(2*numIters, T, n, n)`，`normOut[k]` 为第 k 步归一化结果。
- sumOut 在 x 形状（去掉被求和的那一维）前增加一维 `2*numIters`：逻辑形状为 `(2*numIters, B, S, n)` 或 `(2*numIters, T, n)`，`sumOut[k]` 为第 k 步行/列求和结果。
- **物理存储对齐**：实际 Device 内存中，n 维按 **8 对齐**存储，记 `n_align = ceil(n/8)*8`（n=4/6/8 时 `n_align` **均为 8**）。因此 normOut 物理元素数为 `2*numIters·n·n_align·(B·S 或 T)`，sumOut 物理元素数为 `2*numIters·n_align·(B·S 或 T)`（sumOut 求和后仅保留单个 n 维，按 n_align 对齐）。**当 n=4 时物理元素数是逻辑形状的 2 倍，内存申请须以物理 size 为准。该对齐规则对内存分配与 tiling 切分至关重要。**

## 3. 算子逻辑说明

### 3.1 初始化阶段（第 1 次迭代）

- 对输入 x 沿**最后一维（dim=-1，行方向）**执行 softmax 归一化，使每行元素之和为 1，并加上防除零参数 eps，得到 `normOut[0]`。
- 对 `normOut[0]` 沿倒数第二维（dim=-2，列方向）求和并 keepdim，加上 eps，得到列和 `sumOut[1]`。
- 将 `normOut[0]` 按元素除以 `sumOut[1]`，完成列归一化，得到 `normOut[1]`（此时每列之和为 1）。

### 3.2 交替迭代归一化阶段（第 i 次迭代，i = 1, 2, …, numIters-1）

每次迭代包含一次行归一化和一次列归一化，交替执行使矩阵逐步逼近双随机矩阵：

- **行归一化**：对 `normOut[2i-1]` 沿 dim=-1（行方向）求和并 keepdim，加 eps 得 `sumOut[2i]`；`normOut[2i-1] ÷ sumOut[2i]` 得 `normOut[2i]`（每行和为 1）。
- **列归一化**：对 `normOut[2i]` 沿 dim=-2（列方向）求和并 keepdim，加 eps 得 `sumOut[2i+1]`；`normOut[2i] ÷ sumOut[2i+1]` 得 `normOut[2i+1]`（每列和为 1）。

### 3.3 最终输出与中间结果

- 最终输出取最后一次迭代的归一化结果：`output = normOut[2*numIters-1]`。
- 当 `numIters=1` 时，仅执行初始化阶段，输出 `normOut[1]`（仅满足列和为 1）。
- 可选输出 normOut、sumOut 保存全部迭代中间状态，供反向算子计算梯度使用。

> **注意**：`normOut` 索引从 0 开始（`normOut[0]..normOut[2*numIters-1]` 均有效）；`sumOut` 索引从 **1** 开始（`sumOut[1]` 为首个有效值，由初始化阶段写入），**`sumOut[0]` 为占位未定义**（分配空间但未写入），不作为正确性中间测试点。

## 4. 任务要求

- **精度保障**：针对不同 shape 维度（3 维/4 维）与不同 n 取值（4/6/8），设计算子逻辑，保证 Sinkhorn 迭代归一化精度正确，行和、列和收敛至 1。
- **性能优化**：充分发挥系统带宽能力，合理复用中间结果内存，算子性能更优。
- **切分最优**：探索输入 tensor 在 `(T, n, n)` 与 `(B, S, n, n)` 两种场景下的切分方式，找到不同输入 shape 场景下的最优解。

## 5. 功能示例

```python
import numpy as np

def softmax(x, axis=-1):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def impl(x, eps, num_iters):
    # x: (T, n, n) 或 (B, S, n, n)，最后两维为方阵
    x = x.astype(np.float32)
    # 初始化阶段（第 1 次迭代）
    norm_out = softmax(x, axis=-1) + eps                       # normOut[0]
    sum_out = np.sum(norm_out, axis=-2, keepdims=True) + eps   # sumOut[1]
    norm_out = norm_out / sum_out                              # normOut[1]（列和为1）
    # 交替迭代阶段（i = 1 .. num_iters-1）
    for i in range(1, num_iters):
        sum_out = np.sum(norm_out, axis=-1, keepdims=True) + eps  # sumOut[2i]
        norm_out = norm_out / sum_out                            # normOut[2i]（行和为1）
        sum_out = np.sum(norm_out, axis=-2, keepdims=True) + eps # sumOut[2i+1]
        norm_out = norm_out / sum_out                            # normOut[2i+1]（列和为1）
    return norm_out  # output = normOut[2*num_iters-1]

# 示例1：全 1 矩阵（T=1, n=4），已天然对称，1 次迭代即收敛为均匀双随机矩阵
x = np.ones((1, 4, 4), dtype=np.float32)
y = impl(x, eps=1e-6, num_iters=20)
# squeeze 第0维后矩阵每元素 0.25，行和=1、列和=1

# 示例2：非均匀矩阵（T=1, n=4），迭代 20 次后逼近双随机矩阵
x = np.array([[[4, 3, 2, 1],
               [1, 2, 3, 4],
               [2, 2, 2, 2],
               [3, 1, 4, 2]]], dtype=np.float32)
y = impl(x, eps=1e-6, num_iters=20)
# [[0.5248 0.3838 0.0618 0.0297]
#  [0.0281 0.1517 0.1804 0.6399]
#  [0.2003 0.3981 0.1742 0.2274]
#  [0.2469 0.0664 0.5836 0.1031]]

# 示例3：3 维批量输入（T=2, n=4），对每个方阵独立做 Sinkhorn 变换
```

## 6. 补充：以公开题面接口为准的说明

```
softmax(src, index=None, ptr=None, num_nodes=None, dim=0) -> out
```

参数对应关系：

| 题面 | 评测侧 | 状态 |
|---|---|---|
| `src` | `tensor_x` | 一致 |
| `dim` | `attr_axis` | 含义一致，名称不同 |
| `index` / `ptr` | —— | **缺失，需补充**（二选一的分组参数） |
| 固定 `ε=1e-16` | `attr_eps` | 可传参，题面为固定值 |

**当前建议**：请选手按题面接口实现算子，需要评测侧 cannjudge 补充缺失参数。

---

# 第三题（B组困难题）SparseFlashAttention 算子

## 1. 算子说明

SparseFlashAttention（SFA）是针对长序列推理场景的稀疏注意力计算算子：根据预先选定的稀疏索引，仅对 query 与少量重要的 key/value 位置做注意力计算（而非全部上下文），从而大幅降低计算量。它通常与索引选择算子（如 lightning_indexer）配合使用——后者选出每个 query token 重要的 key 位置索引，本算子依据这些索引完成注意力计算，并针对稀疏访问带来的离散访存做了搬运聚合优化。

### 背景术语（题目内自解释）

- **注意力计算**：标准注意力为 `softmax(Q @ K^T / √d_k) @ V`，其中 Q 为 query、K 为 key、V 为 value，`d_k` 为每个头的维度，`√d_k` 用于缩放防止点积过大。
- **稀疏注意力**：只取少量重要 key/value 位置（记为 K̃, Ṽ）参与计算，公式为 `softmax(Q @ K̃^T / √d_k) @ Ṽ`，降低计算量与访存量。
- **GQA 与 KV 头数**：Grouped Query Attention 中多个 query 头共享一组 key/value 头。本算子 key/value 头数 `KV_N=1`，即所有 query 头共享同一组 KV。
- **RoPE（旋转位置编码）**：将位置信息以旋转矩阵方式注入 query/key，使注意力具备相对位置感知。本算子 query/key 各带一份 RoPE 向量（维度 Dr），**必传不可为空**。
- **MLA-absorb 模式**：Multi-head Latent Attention 的"吸收"模式（`attentionMode=2`），将部分投影矩阵吸收到 query/value 中以减少在线计算量，RoPE 部分单独处理并融合到注意力输出。
- **sinks**：注意力中可学习的偏置项（仅 Ascend 950PR/950DT 支持）。本题目基于 aclnn V1 接口，不涉及该参数。
- **维度符号**：`B`=Batch Size，`Q_S`=query 序列长度，`KV_S`=key/value 序列长度，`Q_N`=query 头数，`KV_N`=key/value 头数（**恒为 1**），`Q_D`/`KV_D`=每个头的维度（**=512**），`Dr`=RoPE 维度（**=64**），`sparse_size`=每个 query token 选取的 key 位置数。
- **数据排布**：本算子**仅支持 BSND 排布**，即 `(B, S, N, D)`。
- **变长序列**：`actual_seq_lengths` 给出每个 batch 实际有效 token 数（超出部分为 padding，不参与计算），为一维长度 B 的 int32 张量；也可传 None 表示与对应序列长度 S 相同。

### 计算公式

```
AttentionOut = softmax( Q @ K̃^T / √d_k ) @ Ṽ
```

其中 K̃, Ṽ 为依据 sparseIndices 从 KV 缓存中离散选取的重要性较高的 Key/Value，`d_k = Q_D`。

## 2. 输入输出说明

### 输入规格

| 输入 | 必选/可选 | 类型 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|---|
| query | 必选 | 张量 | `(B, Q_S, Q_N, Q_D)` | float16/bfloat16 | 输入 Q。`Q_D=512`。不支持空 tensor 与非连续 |
| key | 必选 | 张量 | `(B, KV_S, KV_N, Q_D)` | float16/bfloat16 | 输入 K。`KV_N=1`。不支持空 tensor 与非连续 |
| value | 必选 | 张量 | `(B, KV_S, KV_N, Q_D)` | float16/bfloat16 | 输入 V，shape 与 key 一致。不支持空 tensor 与非连续 |
| sparseIndices | 必选 | 张量 | `(B, Q_S, KV_N, sparse_size)` | int32 | 离散选取 KV 缓存的索引。要求每行有效值在前半部分、无效值在后半部分，`sparse_size>0` |
| actual_seq_lengths_query | 可选 | 张量 | `(B,)` | int32 | 每个 batch 中 query 的有效 token 数。传 None 表示与 `Q_S` 相同 |
| actual_seq_lengths_kv | 可选 | 张量 | `(B,)` | int32 | 每个 batch 中 key/value 的有效 token 数。传 None 表示与 `KV_S` 相同 |
| queryRope | 必选 | 张量 | `(B, Q_S, Q_N, Dr)` | float16/bfloat16 | query 的 RoPE 信息，`Dr=64`。不支持为空 |
| keyRope | 必选 | 张量 | `(B, KV_S, KV_N, Dr)` | float16/bfloat16 | key 的 RoPE 信息，`Dr=64`。不支持为空 |
| scaleValue | 必选 | 属性 | 标量 | float16 | 注意力缩放系数，对应公式中的 `1/√d_k`。**接口传入为 double，内部按 float16 精度处理** |
| sparseBlockSize | 必选 | 属性 | 标量 | int64 | 稀疏选择的块大小：`=1` 为 Token-wise（逐 token 独立选取）；`>1 且 ≤128` 为 Block-wise（块内共享选择决策） |
| sparseMode | 必选 | 属性 | 标量 | int64 | 掩码模式：`0`=全部计算（不屏蔽）；`3`=rightDownCausal（query 序列右端对齐 key 序列右端的下三角掩码）。常用取值 3 |
| attentionMode | 必选 | 属性 | 标量 | int64 | 注意力模式，**仅支持 2**（MLA-absorb 模式） |
| returnSoftmaxLse | 必选 | 属性 | 标量 | bool | 是否输出 `softmaxMaxOut`/`softmaxSumOut`。True 输出、False 不输出；**默认 False** |

说明：属性 `pre_tokens`、`next_tokens` 用于稀疏计算的关联 token 数，**本算子仅支持默认最大值 `2^63-1`**，参赛者无需修改。

### 输出规格

| 输出类型 | 必选/可选 | 形状 | 数据类型 | 含义 |
|---|---|---|---|---|
| attentionOut | 必选 | `(B, Q_S, Q_N, Q_D)` | float16/bfloat16 | 注意力计算最终结果 |
| softmaxMaxOut | 可选 | `(B, KV_N, Q_S, Q_N/KV_N)` | float | 每行 `Q@K̃^T` 的最大值（softmax 数值稳定用）。`returnSoftmaxLse=False` 时不输出 |
| softmaxSumOut | 可选 | `(B, KV_N, Q_S, Q_N/KV_N)` | float | 每行 `Q@K̃^T` 减去 max 后取 exp 的求和（softmax 分母）。`returnSoftmaxLse=False` 时不输出 |

### 形状约束

- 仅支持推理场景，支持图模式。
- **query 头数 Q_N**：Ascend 950PR/950DT 支持 1~128；**Atlas A2/A3 系列仅支持枚举值 1、2、4、8、16、32、64、128（离散取值，非连续范围）**。
- key/value 头数 `KV_N = 1`。
- HeadDim `Q_D = KV_D = 512`，RoPE 维度 `Dr = 64`。
- query、key、value 数据类型**必须一致**。
- RoPE（queryRope/keyRope）**必传，不支持为空**。
- **sparseBlockSize**：Ascend 950PR/950DT 只支持 1；**Atlas A2/A3 系列支持 [1,128] 且为 2 的幂次方**。
- 仅支持 ND 数据格式（任意多维非结构化连续格式）。

### 输出形状计算公式

- `attentionOut`：与 query 形状一致，即 `(B, Q_S, Q_N, Q_D)`。
- `softmaxMaxOut` / `softmaxSumOut`：`(B, KV_N, Q_S, Q_N/KV_N)`，其中 `KV_N=1`，故为 **`(B, 1, Q_S, Q_N)`**，逐 query 头记录 softmax 的最大值与求和。

## 3. 算子逻辑说明

### 3.1 稀疏索引 Gather

对每个 query token，根据 sparseIndices（形状 `(B, Q_S, KV_N, sparse_size)`）中记录的 key 位置索引，从 KV 缓存中离散 gather 出对应的 K̃ 与 Ṽ。要求每行有效索引集中在前半部分、无效值在后半部分，以便聚合访存。`sparseBlockSize` 决定选择粒度：1 为逐 token，>1 为按块共享决策。

### 3.2 注意力主流程

1. 计算 query 与稀疏 key 的相似度并缩放：`score = Q @ K̃^T * scaleValue`（`scaleValue = 1/√d_k`）。
2. 数值稳定 softmax：先取每行最大值 `softmaxMaxOut = max(score)`，再做 `exp(score - max)` 与求和 `softmaxSumOut = Σ exp(...)`，最后归一化得注意力权重 `attn = exp(...) / Σ exp(...)`。
3. 加权求和：`attentionOut = attn @ Ṽ`。

### 3.3 RoPE 与 MLA-absorb 融合

在 `attentionMode=2`（MLA-absorb）模式下，RoPE 采用 **"content 与 rope 沿特征维拼接"** 的方式融合进 score 计算（而非旋转后相加）：

```
score = concat(query, queryRope) @ concat(K̃, keyRopẽ)^T * scaleValue
      = (query @ K̃^T + queryRope @ keyRopẽ^T) * scaleValue
attentionOut = softmax(score) @ Ṽ
```

即 query 的 content 部分（`Q_D=512`）与 queryRope（`Dr=64`）沿特征维拼接为 576 维作为左矩阵，稀疏 gather 出的 K̃ 与 keyRopẽ 同样拼接为 576 维作为右矩阵，做拼接 matmul 得 score。关键点：

- **queryRope/keyRope 已是应用旋转位置编码后的结果**（输入即"位置编码的输出"，算子内**不再做旋转**，直接与 content 拼接）；
- **value（Ṽ）仅用 content 部分（512 维，无 rope）**；
- 数学上等价于 `query@K̃^T + queryRope@keyRopẽ^T`，rope 段承载相对位置信息。

### 3.4 掩码处理

按 `sparseMode` 应用掩码：`sparseMode=0` 不屏蔽；`sparseMode=3`（rightDownCausal）为 query 右端对齐 key 右端的下三角掩码，屏蔽不可见位置使其不参与 softmax。

## 4. 决赛任务要求

- **精度保障**：针对不同 Q_N、变长序列、不同 sparseBlockSize 与 sparseMode 掩码，设计算子逻辑，保证稀疏注意力计算精度正确。
- **性能优化**：针对稀疏索引带来的离散访存，优化 gather 的搬运聚合与 `Q@K̃^T` 矩阵乘的并行度，充分发挥系统带宽与算力。
- **切分最优**：探索 query 序列维度、query 头维度与稀疏索引维度在多 batch / 变长场景下的切分方式，找到不同输入 shape 场景下的最优解。

## 5. 功能示例

```python
import numpy as np

def sparse_flash_attention(query, key, value, sparse_indices, scale_value,
                           act_seq_kv=None):
    # query:(B,Q_S,Q_N,Q_D)  key/value:(B,KV_S,KV_N,Q_D), KV_N=1
    # sparse_indices:(B,Q_S,KV_N,sparse_size)  输出 attentionOut:(B,Q_S,Q_N,Q_D)
    # 为突出稀疏索引选择与注意力主流程，本参考实现聚焦核心公式 softmax(Q@K̃^T/√d)@Ṽ，
    # 省略 3.3 节 RoPE 拼接融合（即 score 中的 queryRope@keyRopẽ^T 段），本示例 score 仅含 query@K̃^T。
    q = query.astype(np.float32); k = key.astype(np.float32); v = value.astype(np.float32)
    B, S1, N1, D = q.shape
    _, S2, N2, _ = k.shape
    out = np.zeros((B, S1, N1, D), dtype=np.float32)
    for b in range(B):
        Klen = S2 if act_seq_kv is None else int(act_seq_kv[b])   # 当前 batch 有效 KV 长度
        K = k[b, :Klen, 0, :]                                     # (Klen, D)
        V = v[b, :Klen, 0, :]                                      # (Klen, D)
        for s in range(S1):
            idx = sparse_indices[b, s, 0]                         # (sparse_size,) 位置索引
            valid = idx[(idx >= 0) & (idx < Klen)]                # 过滤无效索引
            Ksel = K[valid]; Vsel = V[valid]                       # 稀疏 gather 出 K̃, Ṽ
            Q = q[b, s]                                           # (N1, D)
            score = (Q @ Ksel.T) * scale_value                    # (N1, m) 相似度并缩放
            mx = score.max(-1, keepdims=True)                     # softmax 数值稳定
            exp = np.exp(score - mx)
            attn = exp / exp.sum(-1, keepdims=True)               # (N1, m) 注意力权重
            out[b, s] = attn @ Vsel                                # (N1, D) 加权求和
    return out.astype(np.float16)
```

**示例1**：稀疏索引 gather 的作用（BSND，`B=1, Q_S=2, Q_N=2, Q_D=512, KV_S=4, sparse_size=2`）

```python
B, S1, N1, D, S2 = 1, 2, 2, 512, 4
query   = np.ones((B, S1, N1, D), dtype=np.float16)
key     = np.zeros((B, S2, 1, D), dtype=np.float16)
value   = np.tile(np.arange(1, S2+1, dtype=np.float16).reshape(1, S2, 1, 1), (1, 1, 1, D))
sparse_indices = np.array([[[[0, 1]], [[2, 3]]]], dtype=np.int32)  # s0 选 key 0/1，s1 选 key 2/3
scale = 1.0 / np.sqrt(D)
out = sparse_flash_attention(query, key, value, sparse_indices, scale)
# s=0：选 key 0/1（value=1/2），softmax 均匀 => 输出每维均值 1.5
# s=1：选 key 2/3（value=3/4），softmax 均匀 => 输出每维均值 3.5
```

**示例2**：多 batch 变长序列（BSND，`B=2, Q_S=1, Q_N=2, Q_D=512, KV_S=4, sparse_size=2`）

```python
sparse_indices = np.array([[[[0, 1]]], [[[2, 3]]]], dtype=np.int32)  # (2,1,1,2)
out = sparse_flash_attention(query, key, value, sparse_indices, scale, act_seq_kv=[3, 4])
# batch0（有效 KV 长度 3，选 key 0/1）=> 输出均值 1.5
# batch1（有效 KV 长度 4，选 key 2/3）=> 输出均值 3.5
```

---

# 附录：官方题面 vs 真机实测的冲突项

> 官方题面是权威口径，但以下条目**已被真机实测推翻**，实现时**以实测为准**（证据见根目录 `code3.md` §2.4）。

| 项 | 官方题面写 | 真机实测结论 |
|---|---|---|
| `softmaxMaxOut` 的值 | "每行 `Q@K̃^T` 的最大值"、§3.2 写"`softmaxMaxOut = max(score)`"（即**已乘 scale**） | **必须写【未缩放】的行最大**。同一用例 `softmax_sum` 完全正确而 `softmax_max` 恒差 **22.6274 倍 = 1/scale** |

> ⚠️ 该冲突说明：本题"缩放在哪一步完成"极易搞错。早期语义分析（原 `SEMANTICS.md` §7）与设计文档（原 `KERNEL_DESIGN.md` §6）亦沿用旧（已乘 scale）口径，**那两份文档已删除**，勿采信其结论。
