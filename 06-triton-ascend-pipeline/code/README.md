# 代码范围和状态

`upstream/` 保留 Triton-Ascend v3.2.1 固定 commit `2badfc89e70a9b7a5e88463a116c2feddce4b101` 的原始性能指南、load 参数实现、backend 编译选项、multi-buffer 接口/测试及硬件文档。原始路径、版权和 MIT LICENSE 保留；逐文件 URL、SHA-256 见 `source-manifest.json`。

`examples/` 是基于官方指南 `alloc_extend_kernel` Part 2 的独立适配：

- `part2_kernels.py`：整块、循环 tiling、循环 + 无默认 padding 三个 Kernel。
- `benchmark_part2.py`：同一循环 Kernel 再隔离 multi-buffer 开/关，形成四组 NPU 对照。
- `part2_reference.py`：CPU golden，仅定义新增完整页的索引映射。
- `test_reference.py`：7 个边界和尾块语义测试，可在无 NPU 主机执行。
- `device_guard.py` / `test_device_guard.py`：明确接受 A2/A3 的 910B、910_93xx、910C 命名族，拒绝 310、旧/模糊 910、950 和未知名称；无跳过选项。

这些代码不是完整 KV Cache allocator；未处理部分页、KV 状态及模型调度。也没有公开或本机 NPU 性能结论。适配点详见 `../docs/post.md`。

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh cpu-check
# 已准备 Triton-Ascend v3.2.1 / CANN 9.0.0 / A2或A3后：
bash code/reproduce.sh benchmark --output code/work/results.json
```

脚本不会安装包、运行 sudo、下载模型或 CANN。`checkout` 是可选的显式源码下载动作；源码构建安装依赖由使用者先准备。默认 `verify` 完全离线。
