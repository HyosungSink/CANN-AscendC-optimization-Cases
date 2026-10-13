# 离线验证记录

验证日期：2026-10-08（Asia/Shanghai）。主机无 CANN/NPU；未进行设备编译或性能测量。

| 检查 | 结果 |
|---|---|
| 原始源码快照 | 9 个文件的 SHA-256 全部匹配固定 manifest |
| 适配 Python | 所有 examples 文件 AST 解析通过 |
| Shell wrapper | `bash -n code/reproduce.sh` 通过 |
| CPU golden | 7 个边界/尾块参考测试通过 |
| A2/A3 名称白名单 | 2 个测试、15 个接受/拒绝子用例通过 |
| CLI | `benchmark_part2.py --help` 通过，不依赖 torch/Triton/NPU 导入 |

硬件检查现在先明确匹配 `Ascend910B` 型号族、`Ascend910_93xx` 或 `Ascend910C` 型号族。未知名称、310、旧/模糊 910、950 被拒绝；没有绕过开关。名称检查位于取得设备信息后、创建测试输入和启动任何 Kernel 之前。

四组 benchmark 的算法、输入、golden、计时逻辑和编译选项未变。CPU 通过只验证参考映射与入口约束；上板 Kernel 正确性、编译器 multi-buffer 是否生效和真实性能仍未验证。
