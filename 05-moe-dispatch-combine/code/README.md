# 代码范围

`upstream/` 是 sgl-project/sgl-kernel-npu 固定 commit `653e519cf9556b0cc8d9e7966aa4551eb65143ae` 的精选、逐字节原始快照。保留上游相对路径和 MIT LICENSE。`source-manifest.json` 给出每个文件的原始 URL、SHA-256 和大小。

精选内容包括 A3 Dispatch / Combine Kernel 与 Host Tiling、A2 对应及分层 Kernel、Python/C++ 调用桥接、低延迟测试与计时工具、版本/硬件/CI 证据。某些原始入口同时包含 A5 分支，为保持原文件未裁剪；本文和复现脚本只选择 A2/A3。

它是源码阅读快照，不是能够独立构建的完整项目。构建依赖完整上游项目、项目固定的子模块、匹配 CANN 的 PyTorch/torch-npu、CANN Toolkit/ops、驱动和固件。脚本不会 sudo、安装 Python/CANN 包或自动下载子模块。

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh checkout
HARDWARE=A2 bash code/reproduce.sh build
HARDWARE=A2 bash code/reproduce.sh test-int8
HARDWARE=A2 bash code/reproduce.sh test-bf16
```

详情见 `../docs/reproduce.md`。`checkout` 明确联网获取固定源码；其他模式不会隐式联网安装依赖。默认上游工作目录是 `code/work/sgl-kernel-npu`，可用 `SOURCE_DIR` 指定已准备的完整源码。
