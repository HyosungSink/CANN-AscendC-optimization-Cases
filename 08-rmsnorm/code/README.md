# RmsNorm 的原始内核快照

来源为 `cann/ops-nn` 正式 `v9.0.0`，提交 `0413e303e6cd1986b592c3e039f2579ac0f10b62`。`upstream/norm/rms_norm` 保存用于 A2/A3 的 Kernel、Tiling、参数说明、配置与调用/测试代码，所有文件原样复制并保留版权头；`upstream/LICENSE` 为上游 CANN Open Software License Agreement Version 2.0。

快照只收录非 arch35 路径及相关共享文件，未纳入 950 专属 arch35、RegBase 文件和其他设备二进制配置。共享入口中仍可保留条件编译或 Tiling 调度对未收录路径的引用，原文件没有被删行或改写。逐文件范围、排除列表和 SHA256 在 `source-manifest.json`。

**这是源码阅读快照，不能直接作为完整工程构建。** `reproduce.sh` 会获取同一固定提交的完整 ops-nn 仓库，再构建所选单算子。CANN 提供 Ascend C/Reduce/运行时与 opdev 等依赖；完整工程从 CMake 固定 opbase 提交 `ec88904b611b04f9b23b529aee66c78bb7cf6140`。A2 使用 ascend910b，A3 使用 ascend910_93；本机未做 NPU 构建或执行。

```bash
bash reproduce.sh --help
bash reproduce.sh fetch ascend910b
```

默认只 fetch，不执行 CANN 安装、不使用 sudo、不自动安装生成包。实际构建、包加载、精度与计时路线在 [docs/reproduce.md](../docs/reproduce.md)，优化分析在 [docs/post.md](../docs/post.md)。
