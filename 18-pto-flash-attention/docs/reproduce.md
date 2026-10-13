# 复现 PTO Flash Attention 的 preload 对照

Linux，Atlas A2（910B）或 A3（910C），CANN 9.0.0 和匹配驱动/固件。完整源码来自官方 PTO v9.0.0，commit `377526952e95c39f9149cb94d261eba9e394f0d4`。提前准备 NumPy、CMake、bisheng 和 CANN 开发依赖；脚本不会下载安装它们。

## 固定源码与验证

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh checkout
```

默认完整源码在 `code/work/pto-isa`，可用 `SOURCE_DIR` 指定现有固定checkout。`upstream/` 是真实代码和流水图快照，非完整构建依赖。

## 一个合法小 case，两个 preload

```bash
SOC_VERSION=Ascend910B1 QK_PRELOAD=1 bash code/reproduce.sh run
SOC_VERSION=Ascend910B1 QK_PRELOAD=4 bash code/reproduce.sh run
```

固定 case：HEAD=128、S0=128、S1=1024、CUBE_S0=128、CUBE_S1=128、TILE_S1=128、非causal。preload=1在这一tile factor=1规格合法；不能直接把它用于任意更大的TILE_S1。默认NPU_ID=0，SYS_CNT_MULTIPLE=20.0 ns/tick；需按实际计数器环境确认换算。允许的preload为1/2/4。

wrapper 显式生成同一 case 的 header/JSON 和 golden，在原始 `case/build` 目录构建。这个目录选择是有意的：上游 golden 生成器无论 cwd 都写 `case/build`，不能随便把二进制挪到另一目录却保留默认数据路径。脚本不会删除已有 build 目录；改变板型时建议单独checkout，避免混用CMake缓存。

A3请把SOC_VERSION设为板卡实际确认的 `Ascend910_9382` 或 `Ascend910_9391`。wrapper 直接使用 CMake，避免原始 run.sh 的陈旧 SoC regex 限制；这些命令尚未本机上板验证。950不接受。

## 正确性与计时

要求 `test success` 与最终 `o_out` 校验通过；保留 `.csv` report和stdout。原始程序的失败状态需要从输出/报告判断，不能只看shell exit code。

```bash
SOC_VERSION=Ascend910B1 QK_PRELOAD=4 bash code/reproduce.sh profile
```

profile用已安装msprof运行同一case；比较设备事件/流水与内核系统计数器跨度。内置报告取所有block最早start和最晚end，不是平均block时间。shell总时间含构建和生成输入，不能填进性能表。

同一preload至少重复运行多次，记录暖/冷缓存条件。默认上游Host不保证一次输出即是稳定平均值。开启中间量检查会影响执行和数据写回，仅用于定位数值问题。

## 原始数字的限制

原README的归一化TFLOPS乘以24/使用核数，属于外推；实测1/2/4/8核表还改变S0，不是同一工作量强扩展。Host叫gops的值实际除1e6得到MOp；不要把它当GOp换算。测量所用CANN/驱动/历史commit未单列。

本机已做source/evidence hash和wrapper shell检查；没有NPU/CANN设备编译、正确性或性能结果。CPU生成器产生golden也不能替代设备验证。
