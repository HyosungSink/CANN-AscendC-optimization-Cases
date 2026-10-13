# 页面索引生成的 Triton 实际改写：缩小 UB 工作集，去掉无用 padding 依赖

这篇选择 Triton-Ascend 官方性能指南中一个具体改写：`alloc_extend_kernel` 的 **Part 2：新完整页面的物理 token 索引生成**。原始实现一次处理整个 `max_num_extend_tokens`，改写后按 1024 元素循环处理。它比一个向量加法例子更接近实际服务中的间接访存和动态长度任务，但本篇只覆盖一个设备算子片段，不把它写成完整 KV Cache allocator 或端到端调优。

固定版本为 **Triton-Ascend v3.2.1**，commit `2badfc89e70a9b7a5e88463a116c2feddce4b101`。官方 release notes 明确列出 **CANN 9.0.0，Atlas 800 A2 / A3** 配套。Release 同时支持其他硬件，本篇只讨论 A2/A3，排除 950。这个案例有原始优化前后代码，**没有公开的前后性能数字**；本合集提供四个可对照的实现和上板计时脚本，当前状态是“实现与 CPU golden 已检查，NPU benchmark 待运行”。[正式版本配套](https://github.com/triton-lang/triton-ascend/releases/tag/v3.2.1)、[固定原始改写](https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/docs/zh/migration_guide/performance_guidelines.md)。

## 1. 一个 program 负责一条序列，内部需要两种索引

输入包含旧长度 `pre_lens`、新长度 `seq_lens`、空闲物理页编号 `free_page`。每条序列的新增 token 数是 `seq_len - pre_len`；长度的前缀和确定该序列在输出中的起点。新增页数的前缀和确定该序列从空闲页数组的哪个位置取页。

原始 Part 2 处理的是新增区域中的完整页，元素数量为：

```text
num_part2 = floor(seq_len / page_size) × page_size
          - ceil(pre_len / page_size) × page_size
```

对 `offset < num_part2`：

```text
physical_page = free_page[new_page_start + offset // page_size]
physical_token = physical_page × page_size + offset % page_size
out[output_start + offset] = physical_token
```

这里保留的是指南片段定义的输出位置。完整 allocator 还要处理已有部分页、末尾部分页、状态更新和最终布局；这些未包含在原始 Part 2 示例，也未在本篇补造。不同 `free_page` 的物理编号可以不连续，不能把这个任务当成简单连续 copy。

## 2. 原始整块实现的问题：上限决定了局部张量规模

一次性实现创建 `arange(0, max_num_extend_tokens)`，再完成 masked page load、整数除法/取余、索引计算和 store。即使当前序列只新增几百 token，这些局部向量的静态规模也由编译时上限决定。

在 NPU 上，数据搬入、Vector 计算、数据搬出由不同执行单元完成。一个只有一次搬入/计算/搬出的巨大工作块没有多个迭代可供重叠，而且 offsets、mask、page id、计算结果等张量会扩大 UB 工作集。减小每次工作量，既可能降低 UB 压力，也为多个迭代的 MTE2 / Vector / MTE3 重叠提供机会。**代码出现循环只是创造条件，不证明编译器已经生成有效流水。**

## 3. 原始改写：把一次大 load/store 变成循环 tiling

官方指南把 Part 2 改成：

```python
num_loop = tl.cdiv(max_num_extend_tokens, BLOCK_SIZE)
blk_offset = tl.arange(0, BLOCK_SIZE)
for i in range(num_loop):
    offset = blk_offset + i * BLOCK_SIZE
    page = tl.load(free_page_ptr + new_page_start + offset // page_size,
                   mask=offset < num_part2)
    tl.store(out_indices + output_start + offset,
             page * page_size + offset % page_size,
             mask=offset < num_part2)
```

`BLOCK_SIZE=1024` 来自官方示例。每个 tile 覆盖互不相交的 offset 区间，load/store 的有效 mask 相同；因此 Part 2 的有效元素映射保持不变。尾块和没有新完整页的序列由 mask 处理。

本地 [`part2_kernels.py`](../code/examples/part2_kernels.py) 保留单整块、循环整块和去 padding 的三个 Kernel；上板脚本把循环 Kernel 再分别以 `multibuffer=False / True` 启动，得到四组对照：

| 变体 | 单次局部向量 | 自动 multi-buffer | 页面 load 的默认 padding |
|---|---|---|---|
| `single_tile` | 编译上限 | 关闭 | 保留 |
| `loop_no_multibuffer` | TILE | 关闭 | 保留 |
| `loop_multibuffer` | TILE | 开启编译选项 | 保留 |
| `loop_multibuffer_unpadded` | TILE | 开启编译选项 | 去掉 |

固定 backend 的 `AscendOptions` 中存在 `multibuffer` 选项，并传递为自动 multi-buffer 编译参数。它是编译选项，不是性能承诺；资源不足或依赖结构不适合时，不能仅凭选项为 True 就认定流水成功。[固定 backend 代码](https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/third_party/ascend/backend/compiler.py)。

## 4. 为什么去 padding 能减少一个不必要的 V → MTE2 依赖

固定版本的 `tl.load` 支持 `care_padding`。当有 mask、没有显式 `other` 且 `care_padding=True`，NPU 保留默认零填充行为。指南说明这会先由 Vector 设置目标缓冲，再由 MTE2 搬入有效部分，于是搬入要等待 Vector 填充，降低并行空间。

对 Part 2 的 page load，masked lane 只进行逐元素索引计算，并被**相同的 store mask**挡住。它不参与归约，也不影响其他有效 lane，因此可以使用 `care_padding=False`；无效 lane 的值不需要被定义为零。

前缀和那两次 `seq_lens / pre_lens` load 完全不同：masked lane 参与 `tl.sum`，必须保持零值。本地代码明确使用 `other=0`，并保留 padding。不能为了“去同步”给所有 load 都加 `care_padding=False`。

另一个细节来自固定 `core.py`：若显式传了 `other`，`care_padding` **不起作用**。所以页面 load 的无 padding 变体没有同时加 `other=0`；前缀和反而需要 `other=0`。[load 的参数语义](https://github.com/triton-lang/triton-ascend/blob/2badfc89e70a9b7a5e88463a116c2feddce4b101/python/triton/language/core.py)。

## 5. 可验证的等价性，而不是只确认程序没报错

提供的 [`part2_reference.py`](../code/examples/part2_reference.py) 用 CPU 标量方式遍历序列和物理页，生成 golden；它不模拟 Triton 的向量前缀和或 tile 循环。输出先填 `-1`，只有 Part 2 负责的槽位允许改变，比较包括全部 sentinel 槽位。

7 个 CPU 检查覆盖：序列尾部非整页、首部已有部分页、多个序列的输出和空闲页偏移、恰好整页、没有新增完整页、零新增序列、超过 1024 后的非整除尾块及错误输入。上板 benchmark 对四种变体都检查完整输出。随机页编号被打乱，可以发现“把物理页假设为连续编号”的错误。

本地适配还做了两项透明改动：前缀 load 显式 `other=0`；输入和输出使用 int32，以适配有界测试规格。没有把 int64→int32 的收益夹进变体对比，四组使用相同 dtype。真实系统使用 int64 索引时，需要重新确认范围与数学等价，不能直接套用本例。

## 6. 如何得到可信的上板结论

默认测试 batch=8、page_size=16，新增 token 上限分别为 128 / 1025 / 4096，TILE=1024；每组包括短序列、零工作序列与不同旧长度。Triton 的 `arange` 要求合法的编译长度，脚本将上限提升为 2 的幂，并在结果中分别记录逻辑上限和编译上限。

每个变体先编译、运行一次、对 golden，再预热 10 次，使用 NPU Event 记录 30 个样本，报告 median / mean / min / max 和原始样本。输入创建、CPU golden、编译和输出检查都在计时外；缓冲复用，是暖缓存实验，不冒充冷缓存或纯 profiler Kernel duration。仅当同一规格四组都正确时，再看哪一步带来收益。[复现命令](reproduce.md)。

重点观察三件事：单整块是否出现 UB/编译资源问题；循环但关闭 multi-buffer 的延迟是否改变；再开启 multi-buffer、去 padding 是否继续收益。如果 baseline 在某个大规格不能编译，应记录失败原因，不把“失败”换成一个虚构微秒数，也不计算加速比。最终还需用 msprof 的设备流水或编译 IR 确认搬运与计算确实重叠。

**本篇不提供 NPU 成绩：当前主机没有 CANN/NPU。**CPU reference 测试和源码/脚本检查已通过，设备编译、正确性、auto multi-buffer 是否生效和实测收益仍待对应 A2/A3 环境验证。
