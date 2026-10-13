# RotaryPosEmb（RoPE 旋转位置编码）

> 6.1 文档补全阶段产出。developer-doc 依据算子代码（`kernel.asc` / `main.asc`）与设计文档编写，面向使用者。
>
> 契约真值源为本工程根目录 `README.md`（赛题契约）；本文档为其使用侧展开，接口与约束均以已就绪代码为准。

## 功能描述

RotaryPosEmb 实现旋转位置编码（Rotary Position Embedding，RoPE）。RoPE 是 Llama、Qwen、DeepSeek、GLM 等主流大语言模型注入位置信息的核心机制：将 token 向量的最后一维（rope 子维度）按"频率 × 位置"的旋转角两两配对旋转，在不引入可学习参数的前提下编码相对位置关系。

算子完成三步计算：

1. **逆频率**（FP32 计算）：`inv_freq[i] = 1 / theta^(2i/dim)`，`i = 0, 1, ..., dim/2-1`，`dim = rope_dim`；
2. **旋转角与三角函数**（FP32 计算）：`θ[b,s,i] = positions[b,s] · inv_freq[i]`，由 θ 生成 cos/sin（在算子内部完成，cos/sin 不作为输入）；
3. **rotate-half 旋转**（x 最后一维分为前后两半，第 i 对共享旋转角 θᵢ）：

```
y[..., i]         = x[..., i] · cos(θᵢ) − x[..., i + dim/2] · sin(θᵢ)
y[..., i + dim/2] = x[..., i + dim/2] · cos(θᵢ) + x[..., i] · sin(θᵢ)
```

数值实现上，inv_freq 采用标量 double-fp32（pair）算术计算，与 numpy golden 位级对齐（bit-exact）；三角函数先做 Cody-Waite 三项预缩减再调用 2201 高阶 Sin/Cos API，以支持 positions 最大 131072 的大角度输入。工程形态为 Ascend C kernel 直调（`.asc` 直调变体）：`main.asc`（host runner）+ `kernel.asc`（算子 kernel）经 cmake + make 构建为本地可执行 `build/rotary_pos_emb_custom`。

## 接口原型

本算子为 kernel 直调工程（非 custom operator 包），无 torch.ops / aclnn 注册接口。对外契约分三层：

**① 算子计算接口（概念原型）**

```
y = rotary_pos_emb(x, positions, theta=10000.0)
```

**② C 侧 kernel 入口（`kernel.asc`，extern "C"，签名锁定）**

```cpp
void run_kernel(GM_ADDR x, const TensorGroupInfo& info_x,
                GM_ADDR positions, const TensorGroupInfo& info_positions,
                GM_ADDR y, const TensorGroupInfo& info_y,
                int64_t availableCoreNum, aclrtStream stream, float theta);
```

其中张量描述结构（`main.asc` 定义，`kernel.asc` 经 include 复用）：

```cpp
struct TensorInfo {
    const int64_t* shape;   // 各维长度
    int64_t numDims;        // 维数
    int32_t dtype;          // 0=fp32 1=fp16 2=bf16 3=int8 4=int16 5=int32 ...
};
struct TensorGroupInfo {
    const TensorInfo* tensors;
    int64_t numTensors;
};
```

**③ Runner 命令行接口（`main.asc` 可执行入口，`perf/adapter.sh` 契约）**

```bash
./rotary_pos_emb_custom --batch B --seq-len S --num-heads H \
    --rope-dim D --dtype float32|float16 --theta T \
    [--verify|--no-verify] [--case-id ID]
# 无参数运行 → 内置 case0：(1, 7, 1, 16) float32, theta=10000
```

运行约定：工作目录为 `build/`；读取 `input/x.bin`、`input/positions.bin`；写出 `output/y.bin`。选卡遵循 `PERF_DEVICE → ASCEND_RT_VISIBLE_DEVICES` 环境变量约定（runner 内部 `aclrtSetDevice(0)`，可见性由环境变量控制）。

## 参数说明

**算子参数（① ② 层）**

| 参数 | 内存位置 | 方向 | 类型 | 说明 |
|------|----------|------|------|------|
| x | Device（GM） | 输入 | Tensor，float16 / float32，ND | 待旋转的 rope 子维度张量，shape (B, S, H, rope_dim) |
| positions | Device（GM） | 输入 | Tensor，int32，ND | 每个 token 的绝对位置索引，shape (B, S)，元素 ∈ [0, 131072] |
| theta | Host（标量） | 输入（属性） | float，默认 10000.0 | RoPE 几何级数基数，决定各维度频率分布；须为有限正数 |
| y | Device（GM） | 输出 | Tensor，与 x 同 dtype，ND | 旋转位置编码输出，shape / dtype 与 x 严格一致 |
| availableCoreNum | Host（标量） | 输入（launch 参数） | int64_t | 可用 AIV 核数（`ACL_DEV_ATTR_VECTOR_CORE_NUM` 运行时获取，host 侧传入） |
| stream | Host | 输入（launch 参数） | aclrtStream | 同步流 |

**Runner CLI 参数（③ 层）**

| 参数 | 必选 | 默认值 | 说明 |
|------|------|--------|------|
| `--batch B` | 否 | 1 | batch 维 B ∈ [1, 128] |
| `--seq-len S` | 否 | 7 | 序列长度 S ∈ [1, 8192] |
| `--num-heads H` | 否 | 1 | 头数 H ∈ [1, 256] |
| `--rope-dim D` | 否 | 16 | 位置编码维度 ∈ [16, 1024]，且必须为偶数 |
| `--dtype` | 否 | float32 | `float32` 或 `float16`（x 与 y 的 dtype） |
| `--theta T` | 否 | 10000.0 | RoPE 基数，须为有限正数（NaN/Inf/≤0 拒绝） |
| `--verify` / `--no-verify` | 否 | 不校验 | 运行后将 `output/y.bin` 与 `output/golden_y.bin` 按 README §五 阈值比对（镜像 `perf/adapter.sh verify_case`） |
| `--case-id ID` | 否 | 空 | 用例标识（仅打印与记录，不做分派） |
| `-h` / `--help` | — | — | 打印用法 |

**输入文件格式**（`build/` 目录下）

| 文件 | 格式 | 大小 |
|------|------|------|
| `input/x.bin` | 原始二进制，行主序，dtype 由 `--dtype` 决定 | `B·S·H·rope_dim·isize` 字节（fp16 isize=2，fp32 isize=4） |
| `input/positions.bin` | 原始二进制，int32，行主序 | `B·S·4` 字节 |
| `output/y.bin`（写出） | 同 x 格式 | 同 x 大小 |

## 支持数据类型

| 数据类型 | 说明 |
|----------|------|
| float16 | x 与 y 支持（性能主场景，DeepSeek V3 典型） |
| float32 | x 与 y 支持（精度调试场景） |
| int32 | 仅 positions（输入） |
| bfloat16 | 不支持（README §五 阈值表含 bfloat16 列，但接口约束仅声明 float16/float32，视为预留） |

## 约束说明

**Shape / dtype 约束**（runner 侧逐项校验，越界即拒绝退出，不产生 y.bin）

- 维度：4 维输入 (B, S, H, rope_dim)，均为正整数；
  - B（batch）∈ [1, 128]
  - S（序列长度）∈ [1, 8192]
  - H（头数）∈ [1, 256]
  - rope_dim（位置编码维度）∈ [16, 1024]，且必须为偶数
- positions：int32，shape (B, S)，元素 ∈ [0, 131072]（绝对位置索引）；
- theta：有限正数（NaN / Inf / ≤0 拒绝）；
- y：shape / dtype 与 x 严格一致；
- 输入文件必须存在且字节数与 shape 精确匹配（缺失 / 截断均拒绝）；
- bfloat16 不支持。

**精度硬约束**（实现内置，非用户可选项）

- inv_freq 计算与三角函数 cos/sin 生成**必须使用 FP32 中间精度**——positions 最大 131072 × inv_freq 最大 1.0，角度可达 1.3e5 rad，FP32 大参数幅角缩减是精度关键路径；
- inv_freq 采用标量 pair（double-fp32）算术，与 numpy golden 位级对齐（bit-exact，上板验证）；
- fp16 输入时旋转乘加在 FP32 中间域完成后单次舍入回 fp16（CAST_ROUND，与 numpy `astype(float16)` 位级一致）。

**特殊值处理**

- NaN/Inf 输入：输出对应位置为 NaN/Inf，逐位传播，算子不崩溃（校验口径 NaN↔NaN、Inf==Inf 视为匹配）；
- positions = 0：cos=1、sin=0，输出 y == x；
- positions 含超界 / 非法值（防御性）：角度链自然传播为 NaN 输出，不崩溃。

**精度判定阈值**（`--verify` 与评测共用口径）

逐元素通过条件：`abs(actual − golden) ≤ atol + rtol · abs(golden)`；整体双条件：matched_ratio ≥ 0.99 **且** max_abs_error ≤ max_abs_error_limit。

| 数据类型 | atol | rtol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|-----------------------|---------------------|
| float32 | 9.77e-4 | 1.53e-5 | 0.99 | 1e-2 |
| float16 | 1.95e-3 | 1.95e-3 | 0.99 | 1e-1 |

**实现路径说明**（影响性能，不影响正确性）

- kernel 按 `C·isize % 32`（C = rope_dim/2，isize = 2/4 字节）分派快/慢双路径：
  - **快路径**：fp16 且 rope_dim ≡ 0 (mod 32)（如 32/64/96/…/1024），或 fp32 且 rope_dim ≡ 0 (mod 16)（如 16/32/48/…/1024）——整行 head-block 连续搬运 + 合并写出，性能主路径；
  - **慢路径**：其余（如 fp16 的 rope_dim=16/24、fp32 的 rope_dim=24 等）——逐 (row, head) 1D DataCopyPad 搬运，正确性优先，性能不设目标（rope_dim=24 等 32B 非对齐场景）。

**平台 / 环境约束**

- 目标芯片：Ascend910_9362（NpuArch 2201 / DAV_2201 / dav_c220），CANN 9.0.0；
- 构建依赖 `ASCEND_HOME_PATH` 环境变量（未设置时 run.sh / adapter.sh 报错退出，需先 `source <CANN 路径>/set_env.sh`）；
- 编译目标架构默认 `dav-2201`（CMakeLists），可经 `NPU_ARCH` 覆盖（`perf/adapter.sh build` 支持注入）；
- kernel 运行时通过 `PlatformAscendCManager::GetCoreMemSize(UB)` 查询 UB 容量自适应 tiling（不硬编码芯片内存规格）。

## 返回值 / 错误码

run_kernel 无返回值（错误场景输出 stderr 日志后直接返回）。错误语义由 runner 进程退出码承载：

| 返回值 | 含义 |
|--------|------|
| 0 | 运行成功（含 `--verify` 通过；verify 不通过时返回 1） |
| 1 | 参数越界 / dtype 不支持 / theta 非有限正数 / 输入文件缺失或尺寸不匹配 / acl 或内核运行错误 / 校验失败——stderr 输出 `[ERROR] ...` 详情，output/y.bin 不产生（IO 错误场景）或保留 |
| 2 | 用法错误（未知参数、缺少参数值）——stderr 输出用法 |

典型错误输出示例：

```
[ERROR] batch 200 out of range [1, 128]              # 参数越界 → 退出码 1
[ERROR] rope-dim 25 must be even                     # 奇数 rope_dim → 退出码 1
[ERROR] unsupported dtype 'bfloat16' (contract: float16|float32)
[ERROR] input/x.bin missing or size mismatch: expected N bytes, got M
[ERROR] verify FAILED: matched_ratio=0.987654 (<0.99) or max_abs_error=... (>...)
```

## 调用示例

**环境准备**（任一入口都需要）：

```bash
export ASCEND_HOME_PATH=/home/developer/Ascend/cann-9.0.0   # 按实际 CANN 路径
source "${ASCEND_HOME_PATH}/set_env.sh"
```

**示例 1：一键构建 + 运行 + 校验（本地 case0）**

```bash
cd /mnt/workspace/ops/rotaryposemb
bash run.sh
# 流程：source CANN env → cmake + make → python3 scripts/gen_data.py 生成 case0 数据
#      → 运行内置 case0 (1,7,1,16) float32 theta=10000 → verify_result.py 校验
# 输出 "=== PASSED ===" 即通过（120s 超时保护）
```

**示例 2：CLI 直调（DeepSeek V3 典型场景，需先构建）**

```bash
# 构建（增量）：bash perf/adapter.sh build
# 或（全量重建）：bash run.sh 的构建段（cmake + make）

# 准备输入：x (2,512,128,64) fp16，positions (2,512) ∈ [0,8192)
cd build
python3 - <<'EOF'
import numpy as np, sys
sys.path.insert(0, '../scripts')
from RotaryPosEmb import impl
B, S, H, dim = 2, 512, 128, 64
np.random.seed(0)
x = np.random.uniform(-10, 10, (B, S, H, dim)).astype(np.float16)
pos = np.random.randint(0, 8192, (B, S)).astype(np.int32)
x.tofile('input/x.bin'); pos.tofile('input/positions.bin')
impl(x, pos, theta=10000.0).tofile('output/golden_y.bin')   # golden（可选）
EOF

./rotary_pos_emb_custom --batch 2 --seq-len 512 --num-heads 128 \
    --rope-dim 64 --dtype float16 --theta 10000 --verify
# verify: matched_ratio=1.000000, max_abs_error=0 (thresholds: >=0.99, <=0.1)
# verify: PASSED
```

**示例 3：非对齐 rope_dim（慢路径正确性场景）**

```bash
cd build
./rotary_pos_emb_custom --batch 2 --seq-len 16 --num-heads 3 \
    --rope-dim 24 --dtype float16 --theta 10000
# rope_dim=24 非 32B 对齐 → 慢路径（逐 head 1D DataCopyPad），结果仍须过精度校验
```

**示例 4：分级功能测试（68 条用例，L0/L1/L2）**

```bash
bash tests/run_tests.sh --level L0          # 冒烟（评测集 debug case）
bash tests/run_tests.sh --level L1          # 评测集全量 7 case（权威门禁）
bash tests/run_tests.sh --level L2          # runner 异常契约
bash tests/run_tests.sh --list              # 列出用例
# 常用选项：--path A|B|C|all（执行路径）、--case ID（单跑）、--no-build（跳过构建）
# 环境变量：PERF_DEVICE（选卡）、RUNNER_TIMEOUT / PERF_TIMEOUT / L2_TIMEOUT
```

**示例 5：性能采集（共享 runner，仓库根目录）**

```bash
cd /mnt/workspace/ops
python3 scripts/perf.py run --op rotaryposemb --suite quick    # quick 2 case
python3 scripts/perf.py run --op rotaryposemb --suite full     # 全量 7 case
# 采集口径：msprof op，10 warm-up + 50 measured，读 Task Duration(us)
# 注意（已知遗留）：scripts/perf.py:100 的 baseline record/compare 工作流用朴素子串
# 匹配 kernel 名（"rotaryposemb" in "rotary_pos_emb_custom" → False），对本算子不可用；
# 算子侧 tests/perf_collect.py 已用归一化匹配规避，性能对比请走该链路。
```

**等价 Python 参考**（`scripts/RotaryPosEmb.py`，golden 实现，可作为对拍基准）：

```python
import numpy as np

def impl(x, positions, theta=10000.0):
    dim = x.shape[-1]
    x_f = x.astype(np.float32)
    pos_f = positions.astype(np.float32)
    inv_freq = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float32)[: dim // 2] / dim))
    freqs = pos_f[..., None] * inv_freq
    theta_freqs = np.concatenate([freqs, freqs], axis=-1)
    cos = np.cos(theta_freqs)[..., None, :]
    sin = np.sin(theta_freqs)[..., None, :]
    t1, t2 = x_f[..., : dim // 2], x_f[..., dim // 2:]
    y = x_f * cos + np.concatenate([-t2, t1], axis=-1) * sin
    return y.astype(x.dtype)
```

## 支持芯片

| 项目 | 内容 |
|------|------|
| 芯片型号 | Ascend910_9362（短名 Ascend910_93 / Ascend910） |
| 架构 | NpuArch 2201（DAV_2201，dav_c220；CCE_AIV dav-c220-vec） |
| CANN 版本 | 9.0.0（ASCEND_HOME_PATH 指向本版本 toolkit） |
| 计算单元 | AIV（Vector）多核，SIMD / MemBase 编程范式 |

## 附：性能参考

终态实测（`tests/perf_results/perf_p7b_*`，msprof op 采集，3 轮中位数，10 warm-up + 50 measured，Ascend910_9362）：

| Case | Shape | dtype | median |
|------|-------|-------|--------|
| deepseek_v3_2_512_128_64_fp16 | (2,512,128,64) | float16 | 93.84 µs |
| deepseek_v3_2_512_128_64_fp32 | (2,512,128,64) | float32 | 85.76 µs |
| unaligned_2_16_3_24_fp16 | (2,16,3,24) | float16 | 16.50 µs（慢路径，不设性能目标） |
| long_seq_4_8192_128_64_fp16 | (4,8192,128,64) | float16 | 1876.56 µs |
| large_batch_128_8192_8_64_fp16 | (128,8192,8,64) | float16 | 5206.44 µs |
| large_head_dim_2_512_256_1024_fp16 | (2,512,256,1024) | float16 | 1783.40 µs |

相对首轮基线合计 -75.1%（性能迭代 P1/P4/P4b/P7，详见 `.cannbot/rotaryposemb/2.2-开发方案设计.md`「性能迭代修订」节）。瓶颈维度经实测校准为 compute_vector/scalar（fp32 中间域精度硬约束下的向量编排下限），后续版本性能回归以 `perf_p7b_*` 为准入基线。
