# 代码范围

固定正式 tag：`v9.0.0`；commit：`377526952e95c39f9149cb94d261eba9e394f0d4`。`upstream/` 保存 `pto-isa/kernels/manual/a2a3/topk` 的真实 Kernel、Host、构建文件与数据/golden生成代码，以及上游 LICENSE。原文件版权与相对路径保留。逐文件 hash 和固定来源 URL 见 `source-manifest.json`。

它是精选源码快照，不能替代完整构建树。完整源码需要匹配 CANN 9.0.0、驱动/固件、已安装的编译器及算子开发依赖。脚本不会 sudo、安装包、下载 CANN/模型或自动升级依赖；默认 verify 离线。

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh checkout
SOC_VERSION=Ascend910B1 bash code/reproduce.sh run
```

上板复现与计时限制见 `../docs/reproduce.md`。默认完整项目工作目录为 `code/work/pto-isa`，可以用 `SOURCE_DIR` 指定已有固定 checkout。
