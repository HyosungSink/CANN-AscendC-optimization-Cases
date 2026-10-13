# QuantMatmulReluQuant

这是一个 Ascend C 实现的 `INT8 matmul -> FP32 反量化 -> ReLU -> per-row INT8
动态量化` 算子。输入 `x1` 的形状为 `(M,K)`，输入 `x2` 按 `(N,K)` 存储并在
矩阵乘中作为右矩阵转置使用；输出为 `(M,N)` 的 `int8 y` 和 `(M,)` 的
`float32 yScale`。

## 数值语义

```text
acc[m,n] = sum_k int32(x1[m,k]) * int32(x2[n,k])
v[m,n]   = max(0, float(acc[m,n]) * x1Scale[m] * x2Scale[n])
yScale[m] = max(v[m,:]) / 127, or 1.0 when the row is zero
y[m,n]   = saturate(round-to-even(v[m,n] / yScale[m]), int8)
```

`y` 必须 bit-exact；`yScale` 按测试脚本的 `rtol=1e-4, atol=1e-4` 验证。
源码中的 near-half 修正用于保持 FP32 除法和 round-to-even 的边界语义。

## 当前实现

实现位于 [kernel.asc](../code/kernel.asc)，host 测试入口位于 [main.asc](https://gitcode.com/liu-jingzong/cann-1/blob/12f392638497f3f1442f7c46407ef96588cc31f2/main.asc)。
当前硬件为 Ascend910_9362（20 AI Core / 20 Cube / 40 Vector、UB 192 KiB、L1
512 KiB），CANN 9.0.0，编译目标 `dav-2201`。当前 kernel 快照为 commit
`1e3f129`；输入类型为题目规定的 INT8。

`run_kernel` 的调度顺序如下：

1. `sourceK>65535` 时选择 `HugeKKernel`，用八路 INT32 AIV 累加绕过 Matmul tiling
   字段边界。
2. 小 M、短 K、非对齐 N/K 优先选择 `SmallKernel`、`SmallBatch2Kernel` 或
   `SmallK128Batch3Kernel`；物理面板补齐，GM 读写仍按真实尺寸。
3. 其他尺寸使用 `MatmulApiTiling` 生成 `TCubeTiling`，再按形状选择
   `LongColumnStreamKernel`、`ColumnStreamKernel`、`WideKernel` 或 `FusedKernel`。
   AIC 计算 INT32 accumulator，AIV 完成反量化、ReLU、求行最大值和量化。

对于 `N>=16384` 且小 K 非 32 对齐的输入，`PadInputsKernel` 将 A/B 的 K 维补齐到
32 字节友好的长度。host 侧通过 `aclrtMalloc` 分配 accumulator，并在同一分配尾部
放置 padded A/B；kernel 完成后同步并释放。这是当前用户提供方案的一部分，属于额外
设备 workspace，不能再按旧的“无 workspace”文档理解。

## 构建和验证

```bash
cmake --build build -j4
cd build
python3 ../scripts/test_dimensions.py
python3 ../scripts/test_dimensions.py --patterns
python3 ../scripts/test_dimensions.py --boundaries
```

当前本地结果为 15/15 主测试、12/12 patterns、22/22 boundaries 全部通过，主测试
估算均分约 **20.592**（仅用于本地版本比较，不等同线上隐藏形状得分）。逐点耗时和
每轮实验记录见
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md)。

单点测试可以直接传入三元组：

```bash
python3 ../scripts/test_dimensions.py --shape 17 2049 257
```

性能分析命令示例：

```bash
msprof --output=profile_case15 --aic-metrics=PipeUtilization \
  ./quant_matmul_relu_quant_custom 6784 3509 2196
```

当前环境可以返回 kernel 时间，但 `msprof` 不稳定导出 summary；没有 summary 时不应
推断 Cube、Vector 或 MTE 利用率。
