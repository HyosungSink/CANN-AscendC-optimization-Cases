# 复现 PTO 24 核 GEMM

目标环境：Linux、CANN 9.0.0、A2/910B或A3/910C，匹配驱动和固件；已安装bisheng、CMake、NumPy及开发依赖。源码固定官方PTO v9.0.0 commit `377526952e95c39f9149cb94d261eba9e394f0d4`。

## 获取、构建和正确性

```bash
bash code/reproduce.sh verify
bash code/reproduce.sh checkout
SOC_VERSION=Ascend910B1 bash code/reproduce.sh run
```

`SOURCE_DIR`可指向现有完整固定源码。wrapper会在案例目录生成输入/golden，`build-blog`里CMake构建并执行；不下载CANN/模型，不安装包，不删除原上游build目录。6144³矩阵输入和golden会占用数百MB本地空间，CPU golden计算也需要时间。

默认M=K=N=6144、FP16输入、FP32输出、24block、singleCoreM=1536、singleCoreN=1024、baseM=128、baseN=256、baseK=64、stepKa=stepKb=4。Host默认device=0。确认输出`test success`，不是只确认进程正常退出；固定Host在`test failed`时也可能正常返回。

## 性能测量

```bash
SOC_VERSION=Ascend910B1 bash code/reproduce.sh profile
```

profile在正确性执行后，使用已有msprof运行同一二进制。看GEMM设备task duration与MTE/Cube活动。Host没有预热和重复平均机制，因此需要补充同输入多次样本；不把脚本生成数据、编译、ACL初始化、H2D/D2H、校验总时长当Kernel时间。

公开表验证平台是A3 24核，A2时延应独立记录。保存NPU型号/核数、SoC、CANN/驱动/固件、完整commit、安装/构建信息、默认tiling、精度结果、缓存条件、profile输出。公开1.5060ms没有精确历史CANN/commit证据，不设置成当前快照的pass/fail阈值。

## 对照优化

先保留默认源码与结果，再记录修改diff；同步改Host、Kernel模板实例和golden生成参数。可先在相同shape探索stepK，但固定逻辑以stepKa共同触发A/B加载，不能未经检查就把stepKa与stepKb设成不同值。base tile变更必须满足32KiB的单槽L0预算、L1双缓冲和Acc容量。

不同M/N需要同步核网格，默认代码依赖整除。不要把非整除shape当成已经覆盖的规格。对照结果必须保持dtype、布局、B转置/stride、分区和计时边界一致。

本机未运行NPU编译、执行或性能测量，仅完成源码一致性和shell静态检查。
