# 算子调用

## 前提条件

- 环境部署：调用算子之前，请先参考[环境部署](../context/quick_install.md)完成基础环境搭建。

- 调用算子列表：项目可调用的算子参见[算子列表](../op_list.md)，算子对应的aclnn接口参见[aclnn列表](../op_api_list.md)。
- build.sh：算子调用依赖根目录build.sh脚本，可通过`bash build.sh --help`命令查看功能，参数介绍参考[build参数说明](../context/build.md)。

## 源码编译

### 第三方软件依赖

本项目编译过程依赖的第三方开源软件列表如下：

| 开源软件 | 版本 | 下载地址 |
|---|---|---|
| googletest | 1.14.0 | [googletest-1.14.0.tar.gz](https://gitcode.com/cann-src-third-party/googletest/releases/download/v1.14.0/googletest-1.14.0.tar.gz) |
| json | 3.11.3 | [include.zip](https://gitcode.com/cann-src-third-party/json/releases/download/v3.11.3/include.zip) |
| makeself | 2.5.0 | [makeself-release-2.5.0-patch1.tar.gz](https://gitcode.com/cann-src-third-party/makeself/releases/download/release-2.5.0-patch1.0/makeself-release-2.5.0-patch1.tar.gz) |
| pybind11 | 2.13.6 | [pybind11-2.13.6.tar.gz](https://gitcode.com/cann-src-third-party/pybind11/releases/download/v2.13.6/pybind11-2.13.6.tar.gz) |
| eigen | 5.0.0 | [eigen-5.0.0.tar.gz](https://gitcode.com/cann-src-third-party/eigen/releases/download/5.0.0/eigen-5.0.0.tar.gz) |
| protobuf | 25.1.0 | [protobuf-25.1.tar.gz](https://gitcode.com/cann-src-third-party/protobuf/releases/download/v25.1/protobuf-25.1.tar.gz) |
| abseil-cpp | 20230802.1 | [abseil-cpp-20230802.1.tar.gz](https://gitcode.com/cann-src-third-party/abseil-cpp/releases/download/20230802.1/abseil-cpp-20230802.1.tar.gz) |
| opbase(自CANN 9.0.0及以后版本需要下载) | master | [opbase](https://gitcode.com/cann/opbase) |

若您的编译环境可以访问网络，请参考[联网编译](#联网编译)，编译脚本会自动联网下载第三方软件。否则，请参考[未联网编译](#未联网编译)手动下载第三方软件。

准备好开源第三方软件后，可采用如下编译方式，请按需选择：

- **自定义算子包**：

  选择部分算子编译生成的包称为自定义算子包，以**挂载**形式作用于CANN包，不改变原始包内容。生成的自定义算子包优先级高于原始CANN包。该包支持aclnn和图模式调用AI Core算子。

- **ops-transformer包**：

  选择整个项目编译生成的包称为ops-transformer包，可**完整替换**CANN包对应部分。该包支持aclnn和图模式调用AI Core算子。

- **ops-transformer静态库**：

  > 说明：若您需要**基于本项目进行二次发布**并且对**软件包大小有要求**时，建议采用静态库编译，该库可以链接您的应用开发程序，仅保留业务所需的算子，从而实现软件最小化部署。

  表示整个项目编译为一个静态库文件，包含libcann_transformer_static.a和aclnn接口头文件。该包仅支持aclnn调用AI Core算子。

### 联网编译

#### 自定义算子包

1. **编译自定义算子包**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}]
    # 以FlashAttentionScore算子编译为例
    # bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score
    # 编译experimental贡献目录下的算子
    # bash build.sh --pkg --experimental --soc=ascend910b --ops=${experimental_op}
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --vendor_name（可选）：\$\{vendor\_name\}表示构建的自定义算子包名，默认名为custom。
    - --ops（可选）：\$\{op\_list\}表示待编译算子，不指定时默认编译所有算子。格式形如"apply_rotary_pos_emb,rope_quant_kvcache,..."，多算子之间用英文逗号","分隔。
    - --experimental（可选）：表示编译experimental贡献目录下的算子，${experimental_op}为新贡献算子目录名。

    若\$\{vendor\_name\}和\$\{op\_list\}都不传入编译的是ops-transformer包；若编译所有算子的自定义算子包，需传入\$\{vendor\_name\}。当提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-ops-transformer-${vendor_name}_linux-${arch}.run" successfully created.
    ```

    编译成功后，run包存放于项目根目录的build_out目录下。

2. **安装自定义算子包**

    ```bash
    ./cann-ops-transformer-${vendor_name}_linux-${arch}.run
    ```

    自定义算子包安装路径为```${ASCEND_HOME_PATH}/opp/vendors```，\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann。

    > 说明：如果部署算子包时通过配置--install-path参数指定了算子包的安装目录，则在使用自定义算子前，需要执行source \$\{install-path\}/vendors/${vendor_name}/bin/set_env.bash命令，set_env.bash脚本中将自定义算子包的安装路径追加到环境变量ASCEND_CUSTOM_OPP_PATH中，使自定义算子在当前环境中生效。

3. **（可选）删除自定义算子包**

    注意自定义算子包不支持卸载，如需卸载，请删除vendors\/\$\{vendor\_name}目录，并删除vendors/config.ini中load_priority对应\$\{vendor\_name\}的配置项。

#### ops-transformer包

1. **编译ops-transformer包**

    进入项目根目录，执行如下编译命令：

    ```bash
    # 编译除experimental贡献目录外的所有算子
    bash build.sh --pkg --soc=${soc_version}
    # 编译experimental贡献目录下的所有算子
    # bash build.sh --pkg --experimental --soc=${soc_version}
    ```

    - --soc：\$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"，Ascend 950PR/Ascend 950DT产品使用"ascend950"。
    - --experimental（可选）：表示编译experimental贡献目录下的算子。

    若提示如下信息，说明编译成功。

    ```bash
    Self-extractable archive "cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run" successfully created.
    ```

    \$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，run包存放于build_out目录下。

2. **安装ops-transformer包**

    ```bash
    # 安装命令
    ./cann-${soc_name}-ops-transformer_${cann_version}_linux-${arch}.run --full --install-path=${install_path}
    ```

    \$\{install\_path\}：表示指定安装路径，需要与toolkit包安装在相同路径，默认安装在`/usr/local/Ascend`目录。

3. **（可选）卸载ops-transformer包**

    ```bash
    # 卸载命令
    ./${install_path}/cann/share/info/ops_transformer/script/uninstall.sh
    ```

#### ops-transformer静态库

> 说明：静态库仅支持Atlas A2、Atlas A3系列产品。experimental算子暂不支持使用静态库。

1. **编译ops-transformer静态库**

    进入项目根目录，执行如下编译命令：

    ```bash
    bash build.sh --pkg --static --soc=${soc_version}
    ```

    \$\{soc\_version\}表示NPU型号。Atlas A2系列产品使用"ascend910b"（默认），Atlas A3系列产品使用"ascend910_93"。
    若提示如下信息，说明编译并压缩成功。

    ```bash
    [SUCCESS] Build static lib success!
    Successfully created compressed package: ${repo_path}/build_out/cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}.tar.gz
    ```

    \$\{repo\_path\}表示项目根目录，\$\{soc\_name\}表示NPU型号名称，即\$\{soc\_version\}删除“ascend”后剩余的内容。编译成功后，压缩包存放于build_out目录下。

2. **解压ops-transformer静态库**

    进入build_out目录执行解压命令：

    ```bash
    tar -zxvf ./cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}.tar.gz -C ${static_lib_path}
    ```

    \$\{static\_lib\_path\}表示静态库解压路径。解压后目录结构如下：

    ```text
    ├── cann-${soc_name}-ops-transformer-static_${cann_version}_linux-${arch}
    │   ├── lib64
    │   │   ├── libcann_transformer_static.a        # 静态库文件
    │   └── include
    |       ├── ...                                 # aclnn接口头文件
    ```

### 未联网编译

若在没有连接互联网的环境下编译，需要提前准备好依赖的第三方软件，再进行源码编译。具体过程如下：

1. **检查基础环境是否完备**

    请确保已按[环境部署](../context/quick_install.md)完成基础环境搭建，包括CANN包安装、源码下载等。

    - 在联网环境中，进入[本项目主页](https://gitcode.com/cann/ops-transformer)，通过`下载ZIP`或`clone`按钮，根据指导完成源码下载。
    - 连接离线环境，上传源码至您指定的目录下。若下载的是源码压缩包，请先进行解压。

2. **下载第三方软件依赖**
  
    在联网环境中提前下载第三方软件，目前有如下方式，请按需选择：

    - 方式1：根据[第三方软件依赖](#第三方软件依赖)提供的表格手动下载，若从其他地址下载，请确保版本号一致。

    - 方式2：通过[third_lib_download.py](../../../scripts/tools/third_lib_download.py)脚本一键下载，该脚本在本项目`scripts/tools/`目录，下载该脚本并执行如下命令：

        ```bash
        python ${scripts_dir}/third_lib_download.py
        ```

    \$\{scripts\_dir\}表示脚本存放路径，下载的第三方软件包默认存放在当前脚本所在目录。

3. **编译算子包**

   将下载好的第三方软件上传至离线环境，可存放在`third_party`目录或自定义目录下。**推荐前者，其编译命令与联网编译场景下的命令一致。**

    - **third\_party目录**（推荐）
      
      请在本项目根目录创建`third_party`目录（若有则无需创建），将第三方软件拷贝到该指定目录。此时编译命令与联网编译命令一致，具体参考[联网编译](#联网编译)。
      
    - **自定义目录**
      
      在离线环境的任意位置新建`${cann_3rd_lib_path}`目录，将第三方软件拷贝到该目录，请确保该目录有权限访问。

        ```bash
      mkdir -p ${cann_3rd_lib_path}
        ```
      
      此时编译命令需在联网编译命令基础上额外增加`--cann_3rd_lib_path=${cann_3rd_lib_path}`用于指定第三方软件所在路径。假设存放路径为`/path/cann_3rd_lib_path`，不同编译方式对应的命令如下：
      
      - 自定义算子包
      
        ```bash
        bash build.sh --pkg --soc=${soc_version} [--vendor_name=${vendor_name}] [--ops=${op_list}] --cann_3rd_lib_path=${cann_3rd_lib_path}
        # 以FlashAttentionScore算子编译为例
        # bash build.sh --pkg --soc=ascend910b --ops=flash_attention_score --cann_3rd_lib_path=/path/cann_3rd_lib_path
        ```
        
      - ops-transformer整包
      
        ```bash
        bash build.sh --pkg --soc=${soc_version} --cann_3rd_lib_path=${cann_3rd_lib_path}
        # bash build.sh --pkg --soc=ascend910b --cann_3rd_lib_path=/path/cann_3rd_lib_path
        ```
        
      - ops-transformer静态库
      
        ```bash
        bash build.sh --pkg --static --soc=${soc_version} --cann_3rd_lib_path=${cann_3rd_lib_path}
        # bash build.sh --pkg --static --soc=ascend910b --cann_3rd_lib_path=/path/cann_3rd_lib_path
        ```

4. **安装/卸载算子包**
   
    未联网和联网场景下编译得到算子包结果一样，默认存放于项目根目录build_out目录下，并且安装和卸载的操作命令也一样，具体参见[联网编译](#联网编译)。

## 本地验证

通过项目根目录build.sh执行算子和UT用例。目前算子支持API方式（aclnn接口）和图模式调用，**推荐aclnn调用**。

### 执行算子样例

> **说明**：Ascend 950PR产品使用仿真执行算子样例，请见[仿真指导](../debug/op_debug_prof.md#方式二针对ascend-950pr)。

- 基于**自定义算子包**执行算子样例，包安装后，执行如下命令：

    ```bash
    bash build.sh --run_example ${op} ${mode} ${pkg_mode} [--vendor_name=${vendor_name}] [--soc=${soc_version}]
    # 以FlashAttentionScore算子example执行为例
    # bash build.sh --run_example flash_attention_score eager cust --vendor_name=custom
    ```

    - \$\{op\}：表示待执行算子，算子名为小写下划线形式，如flash_attention_score。
    - \$\{mode\}：表示执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
    - \$\{pkg_mode\}：表示包模式，目前仅支持cust，即自定义算子包。
    - \$\{vendor\_name\}（可选）：与构建的自定义算子包设置一致，默认名为custom。
    - \$\{soc_version\}（可选）：表示NPU型号，默认"ascend910b"。当设置为"ascend950"时会运行"arch35"目录下的示例文件。

    说明：\$\{mode\}为graph时，不指定\$\{pkg_mode\}和\$\{vendor\_name\}

- 基于**ops-transformer包**执行算子样例，安装后，执行命令如下：

    ```bash
    bash build.sh --run_example ${op} ${mode} [--soc=${soc_version}]
    # 以FlashAttentionScore算子example执行为例
    # bash build.sh --run_example flash_attention_score eager
    ```

    - \$\{op\}：表示待执行算子，算子名为小写下划线形式，如flash_attention_score。
    - \$\{mode\}：表示算子执行模式，目前支持eager（aclnn调用）、graph（图模式调用）。
    - \$\{soc_version\}（可选）：表示NPU型号，默认"ascend910b"。当设置为"ascend950"时会运行"arch35"目录下的示例文件。

- 基于**ops-transformer静态库**执行算子样例：

    1. **前提条件**

        ops-transformer静态库依赖于ops-legacy静态库和ops-math静态库，将上述静态库准备好，解压并将所有lib64、include目录移动至统一目录\$\{static\_lib\_path\}下。

        > 说明：ops-legacy静态库`cann-${soc_name}-ops-legacy-static_${cann_version}_linux-${arch}.tar.gz`需单击[下载链接](https://ascend.devcloud.huaweicloud.com/artifactory/cann-run-release/software/master)获取， ops-transformer静态库、ops-math静态库暂未提供软件包，请通过本地编译生成。

    2. **创建run.sh**

        在待执行算子`examples\test_aclnn_${op_name}.cpp`同级目录下创建run.sh文件。

        以FlashAttentionScore算子执行test_aclnn_flash_attention_score.cpp为例，示例如下:

        ```bash
        # 环境变量生效
        if [ -n "$ASCEND_INSTALL_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_INSTALL_PATH
        elif [ -n "$ASCEND_HOME_PATH" ]; then
            _ASCEND_INSTALL_PATH=$ASCEND_HOME_PATH
        else
            _ASCEND_INSTALL_PATH="/usr/local/Ascend/cann"
        fi
    
        source ${_ASCEND_INSTALL_PATH}/bin/setenv.bash
    
        # 编译可执行文件
        g++ test_aclnn_flash_attention_score.cpp -I ${static_lib_path}/include -L ${static_lib_path}/lib64 -L ${ASCEND_HOME_PATH}/lib64 -Wl,--allow-multiple-definition \
        -Wl,--start-group -lcann_transformer_static -lcann_math_static -lcann_legacy_static -Wl,--end-group -lgraph -lmetadef \
        -lascendalog -lregister -lopp_registry -lops_base -lascendcl -ltiling_api -lplatform -ldl -lnnopbase -lgraph_base \
        -lc_sec -lunified_dlog -lruntime -lhccl_fwk -o test_aclnn_flash_attention_score   # 替换为实际算子可执行文件名
    
        # 执行程序
        ./test_aclnn_flash_attention_score
        ```

        \$\{static\_lib\_path}表示静态库统一放置路径；\$\{ASCEND\_HOME\_PATH\}已通过环境变量配置，表示CANN toolkit包安装路径，一般为\$\{install\_path\}/cann；最终可执行文件名请替换为实际算子可执行文件名。
        其中lcann\_transformer\_static、lcann\_math\_static、lcann\_legacy\_static表示算子依赖的静态库文件，从静态库统一放置路径\$\{static\_lib\_path\}中获取；lgraph、lmetadef等表示算子依赖的底层库文件，可在CANN toolkit包获取。

    3. **执行run.sh**

        ```bash
        bash run.sh
        ```

无论上述哪种方式，算子样例执行后会打印结果，以FlashAttentionScore算子执行为例：

```text
mean result[0] is: 256.000000
mean result[1] is: 256.000000
mean result[2] is: 256.000000
mean result[3] is: 256.000000
mean result[4] is: 256.000000
mean result[4] is: 256.000000
...
mean result[65532] is: 256.000000
mean result[65533] is: 256.000000
mean result[65534] is: 256.000000
mean result[65535] is: 256.000000
```

### 执行算子UT

> 说明：执行UT用例依赖googletest单元测试框架，详细介绍参见[googletest官网](https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests)。

```bash
# 安装根目录下test相关requirements.txt依赖
pip3 install -r tests/requirements.txt
# 方式1: 编译并执行指定算子和对应功能的UT测试用例（选其一）
bash build.sh -u --[opapi|ophost|opkernel] --ops=flash_attention_score
# 方式2: 编译并执行所有的UT测试用例
# bash build.sh -u
# 方式3: 编译所有的UT测试用例但不执行
# bash build.sh -u --noexec
# 方式4: 编译并执行对应功能的UT测试用例（选其一）
# bash build.sh -u --[opapi|ophost|opkernel]
# 方式5: 编译对应功能的UT测试用例但不执行（选其一）
# bash build.sh -u --noexec --[opapi|ophost|opkernel]
# 方式6: 执行UT测试用例时可指定soc编译
# bash build.sh -u --[opapi|ophost|opkernel] [--soc=${soc_version}]
```

如需验证ophost功能是否正常，执行如下命令：

```bash
bash build.sh -u --ophost
```

执行完成后出现如下内容，表示执行成功。

```bash
Global Environment TearDown
[==========] ${n} tests from ${m} test suites ran. (${x} ms total)
[  PASSED  ] ${n} tests.
[100%] Built target transformer_op_host_ut
```

\$\{n\}表示执行了n个用例，\$\{m\}表示m项测试，\$\{x\}表示执行用例消耗的时间，单位为毫秒。
