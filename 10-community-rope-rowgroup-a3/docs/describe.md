# 【初赛】RotaryPosEmb算子

## 一、赛题背景
旋转位置编码（Rotary Position Embedding，RoPE）是现代大语言模型注入位置信息的核心机制，被Llama、Qwen、DeepSeek、GLM等主流模型广泛采用。RoPE将token向量的维度配对为二维平面上的点，并按"频率 × 位置"的角度对每对维度做旋转，从而在不引入可学习参数的前提下编码相对位置关系。

本题要求基于PyTorch中旋转位置编码的核心业务逻辑，采用Ascend C编程语言进行算子原生开发，在昇腾NPU硬件上实现一款高性能的RotaryPosEmb算子。

## 二、算子功能描述
实现的RotaryPosEmb算子需完成以下核心计算：
1. 根据维度与theta计算逆频率 $\operatorname{inv\_freq}[i] = 1 / \operatorname{theta}^{(2i/\operatorname{dim})}$
2. 由位置索引与逆频率生成旋转角，进而计算cos/sin
3. 对输入张量施加旋转位置编码：将x的最后一维配对后按对应旋转角旋转，得到输出y

算子输入的x为解耦出来的位置编码子维度（rope部分），只对rope_dim做旋转。

## 三、核心定义与约束
### 3.1 参考算子
PyTorch中旋转位置编码的一种等价实现如下。旋转的具体实现方式（rotate‑half、复数乘法或其他数学等价方法）不限，以下实现仅用于说明数学逻辑：
```python
import torch

def apply_rotary_pos_emb(x, positions, theta):
    dim = x.shape[-1]
    # 1. 逆频率
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))

    # 2. 旋转角与 cos/sin
    freqs = positions.float().unsqueeze(-1) * inv_freq      # [B, S, dim/2]
    theta_freqs = torch.cat([freqs, freqs], dim=-1)        # [B, S, dim]
    cos = torch.cos(theta_freqs).unsqueeze(2)              # [B, S, 1, dim]
    sin = torch.sin(theta_freqs).unsqueeze(2)

    # 3. rotate‑half 旋转
    def rotate_half(t):
        t1, t2 = t[..., : dim // 2], t[..., dim // 2 :]
        return torch.cat((-t2, t1), dim=-1)

    return x * cos + rotate_half(x) * sin
```

### 3.2 数学公式
设输入 $x$ 的最后一维为 $\operatorname{dim}$（rope_dim），位置索引为 $p$（来自positions），基础逆频率：
$$
\operatorname{inv\_freq}[i] = \theta^{-2i/\operatorname{dim}},\quad i=0,1,\dots,\operatorname{dim}/2-1
$$
旋转角与三角函数：
$$
\Theta[b,s] = \operatorname{concat}\big(\ p[b,s]\cdot\operatorname{inv\_freq}[i],\ p[b,s]\cdot\operatorname{inv\_freq}[i]\ \big)
$$
$$
\cos = \cos(\Theta),\quad \sin = \sin(\Theta)
$$

旋转位置编码（x的最后一维分成前后两半，前$\operatorname{dim}/2$维与后$\operatorname{dim}/2$维两两配对，第$i$对共享旋转角$\theta_i = p[b,s]\cdot\operatorname{inv\_freq}[i]$）：
$$
y[\dots,i] = x[\dots,i]\cos\theta_i - x[\dots,i+\operatorname{dim}/2]\sin\theta_i
$$
$$
y[\dots,i+\operatorname{dim}/2] = x[\dots,i+\operatorname{dim}/2]\cos\theta_i + x[\dots,i]\sin\theta_i
$$
其中 $i = 0,1,\dots,\operatorname{dim}/2-1$。旋转的具体实现方式不限，任何数学等价方法（如rotate‑half、复数乘法等）均可。

### 3.3 输入输出与属性总览
| 类型 | 参数名 | 类型 | 维度形状 | 支持数据类型 | 数据格式 | 备注 |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| INPUT(必选) | x | tensor | (B, S, H, rope_dim) | float16, float32 | ND | 待旋转的rope子维度张量 |
| INPUT(必选) | positions | tensor | (B, S) | int32 | ND | 每个token的位置索引 |
| ATTR(属性) | theta | float | - | - | - | RoPE基础频率基数，默认10000.0 |
| OUTPUT(输出) | y | tensor | 与x一致 | 与x一致 | 与x一致 | 旋转位置编码后的输出 |

### 3.4 关键输入约束
- 数据类型: float16, float32
- 维度场景: 4维输入 $(B, S, H, \operatorname{rope\_dim})$，B为batch，S为序列长度，H为头数，rope_dim为位置编码维度

**维度取值范围（均为正整数）**：
- B（batch）: $[1,128]$
- S（序列长度）: $[1,8192]$
- H（头数）: $[1,256]$
- rope_dim（位置编码维度）: $[16,1024]$，且必须为偶数

**位置索引约束**: positions中每个元素为 $[0,131072]$ 的整数，表示绝对位置

**典型shape**:
- DeepSeek V3实际场景: $(B, S, 128, 64),\ \operatorname{rope\_dim}=64,\ 128$ 个head
- 小规模调试: $(1, 8, 1, 16)$

**非对齐场景**: rope_dim、H、S的任意组合可能导致非对齐的内存访问，算子需正确处理

### 3.5 核心属性说明
- theta（float，默认10000.0）: RoPE的几何级数基数，决定各维度频率的分布范围

### 3.6 输出严格要求
- 形状约束: $\boldsymbol{y}.\operatorname{shape} == \boldsymbol{x}.\operatorname{shape}$，所有维度尺寸不变
- 类型约束: y的数据类型与x一致
- 数值等价: 输出与PyTorch参考实现在合理浮点误差范围内一致
- 中间精度要求: inv_freq计算、三角函数cos/sin生成必须使用FP32精度，避免长序列下角度误差累积

### 3.7 特殊值处理规则
- NaN/Inf输入: 输入含NaN/Inf时，输出对应位置为NaN/Inf，算子不应崩溃
- 位置为0: positions为0时，$\cos=1、\sin=0$，输出 $y == x$（旋转角为0）
- rope_dim边界: rope_dim为偶数，旋转配对的前后两半各为$\operatorname{dim}/2$维

## 四、规则要求
1. cos/sin生成规则: 逆频率计算与三角函数cos/sin的生成在算子内部完成，cos/sin不作为算子输入
2. 实现方式规则: 旋转位置编码的具体实现方式不限（如rotate‑half、复数乘法等），任何与数学定义等价的实现均可
3. 中间精度规则: 逆频率与三角函数计算必须使用FP32精度，旋转乘法可在输入精度下进行，需保证最终输出与golden的误差在阈值内

## 五、精度判断规则
以更高精度的参考实现结果作为golden（标杆）进行逐元素比较。

逐元素通过条件：
$$
\operatorname{abs}(\operatorname{actual} - \operatorname{golden}) \le \operatorname{atol} + \operatorname{rtol} \times \operatorname{abs}(\operatorname{golden})
$$
其中 $\operatorname{atol}$ 为绝对误差，$\operatorname{rtol}$ 为相对误差。

整体通过条件：通过率 $\operatorname{matched\_ratio} = \text{通过元素数} \;/\; \text{总元素数}$，当同时满足以下条件时判定用例通过：
1. $\operatorname{matched\_ratio} \ge \operatorname{required\_matched\_ratio}$
2. $\operatorname{max\_abs\_error} \le \operatorname{max\_abs\_error\_limit}$（$\operatorname{max\_abs\_error}$ 为任意元素的最大绝对误差）

误差阈值表：

| 数据类型 | float32 | float16 | bfloat16 |
| ---- | ---- | ---- | ---- |
| atol | 9.77e‑4 | 1.95e‑3 | 1.56e‑2 |
| rtol | 1.53e‑5 | 1.95e‑3 | 1.56e‑2 |
| required_matched_ratio | 0.99 | 0.99 | 0.99 |
| max_abs_error_limit | 1e‑2 | 1e‑1 | 1e‑0 |

## 六、示例说明
**示例1: 极小规模调试**
- 输入x: shape=$[1, 8, 1, 16]$, dtype=float32
- positions: shape=$[1, 8]$, 值为0~7的连续位置
- 属性: theta=10000.0
- 输出y: shape=$[1, 8, 1, 16]$，为x按各自位置旋转后的结果

**示例2: DeepSeek V3实际场景**
- 输入x: shape=$[2, 512, 128, 64]$, dtype=float16
- positions: shape=$[2, 512]$, 值为$[0,8192)$的随机位置
- 属性: theta=10000.0
- 输出y: shape=$[2, 512, 128, 64]$

**示例3: 非对齐rope_dim**
- 输入x: shape=$[2, 16, 3, 24]$, dtype=float16（rope_dim=24 非2的幂）
- positions: shape=$[2, 16]$
- 属性: theta=10000.0
- 输出y: shape=$[2, 16, 3, 24]$