# INT8 per-token反量化的scale预计算与CV流水：原始源码与复现入口

范围：BmmDequantPertokenOpt：小M、50MiB workspace限制、scale外积预计算、15次分组同步。

快照来自官方 ops-nn 正式 `v9.0.0`，提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`；保存了 74 份上游原文件（约 1254.6 KiB），包括实际 Kernel、Host Tiling / 注册、API实现和指定调用样例。核心文件没有改写为伪代码。保留上游原始版权头和 [LICENSE](upstream/LICENSE)。

`source-manifest.json` 逐文件记录路径、SHA256、大小与固定提交URL。A2/A3注册证据位于 `matmul/quant_batch_matmul_v3/op_host/quant_batch_matmul_v3_def.cpp` 第 76–77 行。

快照主动排除950的arch35、310P/arch20、测试、配置知识库等；共享入口仍有条件include或其他模板，不能把这个阅读快照当成独立编译工程。构建必须用完整固定版本仓库、实际CMake指定的opbase依赖和CANN Toolkit。

- [文章](../docs/post.md)：实际数据切分、搬运、同步与取舍。
- [复现](../docs/reproduce.md)：源码fetch、离线依赖、单算子自定义包构建和样例运行。
- [来源](../docs/sources.md)、[机器可读元数据](../docs/metadata.json)。

`bash reproduce.sh verify` 离线校验快照；`fetch`获取固定源码；`build`要求Linux+CANN 9.0.0并已准备离线依赖；`run-example`要求已经装入自定义包。wrapper不安装CANN、不sudo、不自动下载大二进制包。

样例边界：示例文件名明确面向A2/A3；模板选择依赖pertoken、输出dtype、shape、workspace等条件。调用成功不等于已运行opt模板，必须检查tiling/profiler。

本机macOS、无NPU，只完成静态源码与脚本检查，未声称NPU编译、数值或性能测试通过。
