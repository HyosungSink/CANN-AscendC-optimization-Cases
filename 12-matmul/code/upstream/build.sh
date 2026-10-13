#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ============================================================================

set -e
RELEASE_TARGETS=("ophost" "opapi" "onnxplugin" "opgraph")

SUPPORT_COMPUTE_UNIT_SHORT=("ascend031" "ascend035" "ascend310b" "ascend310p" "ascend910_93" "ascend950" "ascend910b" "ascend910" "kirinx90" "kirin9030" "mc62cm12a")
declare -A SOC_TO_ARCH
SOC_TO_ARCH=(["ascend310b"]="3002" ["ascend310p"]="2002" ["ascend910_93"]="2201" ["ascend910b"]="2201"
            ["ascend950"]="3510" ["ascend910"]="1001" ["mc62cm12a"]="5102")
# 对SUPPORT_COMPUTE_UNIT_SHORT按字符串长度从长到短排序，避免前缀匹配时出错
SUPPORT_COMPUTE_UNIT_SHORT=($(printf '%s\n' "${SUPPORT_COMPUTE_UNIT_SHORT[@]}" | awk '{print length($0) " " $0}' | sort -rn | cut -d ' ' -f2-))
TRIGER_UTS=()

# 所有支持的短选项
SUPPORTED_SHORT_OPTS="hj:vO:uf:-:"

# 所有支持的长选项
SUPPORTED_LONG_OPTS=(
  "help" "ops=" "soc=" "vendor_name=" "build-type=" "cov" "noexec" "noaicpu" "opkernel" "opkernel_aicpu" "opkernel_aicpu_test" "static"
   "jit" "pkg" "asan" "make_clean_all" "make_clean" "no_force"
  "ophost" "opgraph" "opapi" "run_example" "example_name=" "genop=" "genop_aicpu=" "experimental" "cann_3rd_lib_path=" "oom" "onnxplugin" "dump_cce"
  "simulator" "bisheng_flags=" "kernel_template_input=" "module_extension=" "noaclnn"
)

source "./install_deps.sh"

set +u
if ! check_dependencies_silent "$@"; then
  exit 1
fi

in_array() {
  local needle="$1"
  shift
  local haystack=("$@")
  for item in "${haystack[@]}"; do
    if [[ "$item" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

# 检查参数是否合法
check_option_validity() {
  local arg="$1"

  # 检查短选项
  if [[ "$arg" =~ ^-[^-] ]]; then
    local opt_chars=${arg:1}

    # 获取需要参数的短选项列表
    local needs_arg_opts=$(echo "$SUPPORTED_SHORT_OPTS" | grep -o "[a-zA-Z]:" | tr -d ':')

    # 逐个检查短选项字符
    local i=0
    while [ $i -lt ${#opt_chars} ]; do
      local char="${opt_chars:$i:1}"

      # 检查是否为支持的短选项
      if [[ ! "$SUPPORTED_SHORT_OPTS" =~ "$char" ]]; then
        print_error "Invalid short option: -$char"
        return 1
      fi

      # 如果这个选项需要参数，跳过参数部分
      if [[ "$needs_arg_opts" =~ "$char" ]]; then
        # 跳过参数部分（可能是数字或其他字符）
        while [ $i -lt ${#opt_chars} ] && [[ "${opt_chars:$i:1}" =~ [0-9a-zA-Z] ]]; do
          i=$((i + 1))
        done
      else
        i=$((i + 1))
      fi
    done
    return 0
  fi

  # 检查长选项
  if [[ "$arg" =~ ^-- ]]; then
    local long_opt="${arg:2}"
    local opt_name="${long_opt%%=*}"

    # 检查是否在支持的长选项列表中
    for supported_opt in "${SUPPORTED_LONG_OPTS[@]}"; do
      # 处理带=的长选项
      if [[ "$supported_opt" =~ =$ ]]; then
        local base_opt="${supported_opt%=}"
        if [[ "$opt_name" == "$base_opt" ]]; then
          return 0
        fi
      else
        # 处理不带=的长选项
        if [[ "$opt_name" == "$supported_opt" ]]; then
          return 0
        fi
      fi
    done

    print_error "Invalid long option: --$opt_name"
    return 1
  fi

  # 如果不是选项，可能是其他参数
  return 0
}

dotted_line="----------------------------------------------------------------"
export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
REPOSITORY_NAME="nn"

CORE_NUMS=$(cat /proc/cpuinfo | grep "processor" | wc -l)
ARCH_INFO=$(uname -m)


export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ACLNN_INCLUDE_PATH="${INCLUDE_PATH}/aclnn"
export COMPILER_INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export GRAPH_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/graph"
export EXTERNAL_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/external"
export GE_INCLUDE_PATH="${COMPILER_INCLUDE_PATH}/ge"
export INC_INCLUDE_PATH="${ASCEND_OPP_PATH}/built-in/op_proto/inc"
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/include"
export EAGER_LIBRARY_OPP_PATH="${ASCEND_OPP_PATH}/lib64"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export GRAPH_LIBRARY_STUB_PATH="${ASCEND_HOME_PATH}/lib64/stub"
export GRAPH_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
CANN_3RD_LIB_PATH="${BUILD_PATH}/third_party"
# print usage message
usage() {
  local specific_help="$1"

  if [[ -n "$specific_help" ]]; then
    case "$specific_help" in
      pkg)
        echo "pkg Build Options:"
        echo $dotted_line
        echo "    --pkg                  Build run pkg with kernel bin"
        echo "    --jit                  Build run pkg without kernel bin"
        echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
        echo "    --vendor_name=name     Specify custom operator pkg vendor name"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    -j[n]                  Compile thread nums, default is 8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3]"
        echo "    --asan                 Enable ASAN (Address Sanitizer) on the host side"
        echo "    --build-type=<TYPE>    Specify build type (TYPE options: Release/Debug), Default: Release"
        echo "    --experimental         Build experimental version"
        echo "    --cann_3rd_lib_path=<PATH>"
        echo "                           Set ascend third_party package install path, default ./third_party"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo "    --dump_cce             Dump kernel precompiled files (.i) for debugging"
        echo "    --bisheng_flags=ccec_g,oom"
        echo "                           Specify bisheng compiler flags (comma-separated for multiple)"
        echo "    --kernel_template_input="args0=args0;args1=args1;args2=args2;args3=args3""
        echo "                           Specify kernel template input arguments (semicolon-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --pkg --soc=ascend910b --vendor_name=customize -j16 -O3"
        echo "    bash build.sh --pkg --ops=transpose_batch_mat_mul,fatrelu_mul --build-type=Debug"
        echo "    bash build.sh --pkg --soc=ascend910b --ops=transpose_batch_mat_mul --oom"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=\${experimental_op}"
        echo "    bash build.sh --pkg --experimental --soc=ascend910b --ops=\${experimental_op} --bisheng_flags=ccec_g,oom"
        echo "    bash build.sh --pkg --soc=ascend910b --ops=transpose_batch_mat_mul --kernel_template_input="BATCH_SPLIT=0;PP_MAT_MUL_EINSUM_MODE=0;PERM_X1=2;PERM_X2=0""
        return
        ;;
      opkernel)
        echo "Opkernel Build Options:"
        echo $dotted_line
        echo "    --opkernel             Build binary kernel"
        echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo "    --dump_cce             Dump kernel precompiled files (.i) for debugging"
        echo "    --bisheng_flags=ccec_g,oom"
        echo "                           Specify bisheng compiler flags (comma-separated for multiple)"
        echo "    --no_force             Don't force dependency installation"
        echo "    --kernel_template_input="args0=args0;args1=args1;args2=args2;args3=args3""
        echo "                           Specify kernel template input arguments (semicolon-separated for multiple)"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=transpose_batch_mat_mul,fatrelu_mul"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=transpose_batch_mat_mul,fatrelu_mul --build-type=Debug"
        echo "    bash build.sh --opkernel --soc=ascend310p --ops=transpose_batch_mat_mul,fatrelu_mul --oom"
        echo "    bash build.sh --opkernel --soc=ascend910b --ops=transpose_batch_mat_mul --kernel_template_input="BATCH_SPLIT=0;PP_MAT_MUL_EINSUM_MODE=0;PERM_X1=2;PERM_X2=0""
        return
        ;;
      opkernel_aicpu)
        echo "AICPU Opkernel Build Options:"
        echo $dotted_line
        echo "    --opkernel_aicpu       Build AICPU kernel"
        echo "    --soc=soc_version      Compile for specified Ascend SoC (comma-separated for multiple)"
        echo "    --ops=op1,op2,...      Compile specified operators (comma-separated for multiple)"
        echo "    --build-type=<Type>    Specify build-type (Type options: Release/Debug), Default:Release"
        echo "    --oom                  Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=transpose_batch_mat_mul,fatrelu_mul"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=transpose_batch_mat_mul,fatrelu_mul --build-type=Debug"
        echo "    bash build.sh --opkernel_aicpu --soc=ascend910b --ops=transpose_batch_mat_mul,fatrelu_mul --oom"
        return
        ;;
      test)
        echo "Test Options:"
        echo $dotted_line
        echo "    -u                     Build and run all unit tests"
        echo "    --noexec               Only compile ut, do not execute"
        echo "    --asan                 Enable ASAN (Address Sanitizer) on the host side"
        echo "    --ophost -u            Same as ophost test"
        echo "    --opgraph -u           Same as opgraph test"
        echo "    --opapi -u             Same as opapi test"
        echo "    --opkernel -u          Same as opkernel test"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh -u"
        echo "    bash build.sh -u --ophost"
        return
        ;;
      clean)
        echo "Clean Options:"
        echo $dotted_line
        echo "    --make_clean_all       Clean all build artifacts and related files"
        echo "    --make_clean           Clean build artifacts"
        echo $dotted_line
        return
        ;;
      ophost)
        echo "Ophost Build Options:"
        echo $dotted_line
        echo "    --ophost               Build ophost library"
        echo "    -j[n]                  Compile thread nums, default is 8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3]"
        echo "    --build-type=<TYPE>    Specify build type (TYPE options: Release/Debug), Default: Release"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --ophost -j16 -O3"
        echo "    bash build.sh --ophost --build-type=Debug"
        return
        ;;
      onnxplugin)
        echo "ONNXPlugin Build Options:"
        echo $dotted_line
        echo "    --onnxplugin           Build onnxplugin library"
        echo "    -j[n]                  Compile thread nums, default is 8, eg: -j8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3], eg:-O3"
        echo "    --debug                Build with debug mode"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --onnxplugin -j16 -O3"
        echo "    bash build.sh --onnxplugin --debug"
        return
        ;;
      opgraph)
        echo "Opgraph Build Options:"
        echo $dotted_line
        echo "    --opgraph              Build opgraph library"
        echo "    -j[n]                  Compile thread nums, default is 8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3]"
        echo "    --debug                Build with debug mode"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opgraph -j16 -O3"
        echo "    bash build.sh --opgraph --debug"
        return
        ;;
      opapi)
        echo "Opapi Build Options:"
        echo $dotted_line
        echo "    --opapi                Build opapi library"
        echo "    -j[n]                  Compile thread nums, default is 8"
        echo "    -O[n]                  Compile optimization options, support [O0 O1 O2 O3]"
        echo "    --build-type=<TYPE>    Specify build type (TYPE options: Release/Debug), Default: Release"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --opapi -j16 -O3"
        echo "    bash build.sh --opapi --build-type=Debug"
        return
        ;;
      run_example)
        echo "Run example Options:"
        echo $dotted_line
        echo "    --run_example op_name  mode[eager:graph] [pkg_mode --vendor_name=name --example_name=name --soc=soc_version]      Compile and execute the test_aclnn_xxx.cpp/test_geir_xxx.cpp"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --run_example mat_mul_v3 eager"
        echo "    bash build.sh --run_example mat_mul_v3 eager --soc=ascend950"
        echo "    bash build.sh --run_example mat_mul_v3 graph"
        echo "    bash build.sh --run_example mat_mul_v3 eager --example_name=mm"
        echo "    bash build.sh --run_example mat_mul_v3 eager cust"
        echo "    bash build.sh --run_example mat_mul_v3 eager cust --vendor_name=custom"
        echo "    bash build.sh --run_example mat_mul_v3 eager cust --vendor_name=custom --simulator"
        return
        ;;
      genop)
        echo "Gen Op Directory Options:"
        echo $dotted_line
        echo "    --genop=op_class/op_name      Create the initial directory for op_name undef op_class"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --genop=op_class/op_name"
        return
        ;;
      genop_aicpu)
        echo "Gen Op Directory Options:"
        echo $dotted_line
        echo "    --genop_aicpu=op_class/op_name      Create the initial directory for op_name undef op_class"
        echo $dotted_line
        echo "Examples:"
        echo "    bash build.sh --genop_aicpu=op_class/op_name"
        return
        ;;
    esac
  fi
  echo "build script for ops-nn repository"
  echo "Usage:"
  echo "    bash build.sh [-h] [-j[n]] [-v] [-O[n]] [-u] "
  echo ""
  echo ""
  echo "Options:"
  echo $dotted_line
  echo "    Build parameters "
  echo $dotted_line
  echo "    --help, -h Print usage"
  echo "    -j[n] Compile thread nums, default is 8"
  echo "    -v Cmake compile verbose"
  echo "    -O[n] Compile optimization options, support [O0 O1 O2 O3]"
  echo "    -u Compile all ut, default run ophost opgraph opapi test"
  echo $dotted_line
  echo "    example, Build ophost test with O0 level compilation optimization and do not execute."
  echo "    ./build.sh -u --ophost --noexec -O0 -j8"
  echo $dotted_line
  echo "    The following are all supported arguments:"
  echo $dotted_line
  echo "    --build-type=<TYPE> Specify build type (TYPE options: Release/Debug), Default: Release"
  echo "    --noexec Only compile ut, do not execute the compiled executable file"
  echo "    --make_clean make clean"
  echo "    --make_clean_all make clean and delete related file"
  echo "    --asan enable asan on the host side"
  echo ""
  echo "    --ops Compile specified operator, use snake name, like: --ops=add,add_lora, use ',' to separate different operator"
  echo "    --soc Compile binary with specified Ascend SoC, like: --soc=ascend910b"
  echo "    --vendor_name Specify the custom operator pkg vendor name, like: --vendor_name=customize, default to customize-nn"
  echo "    --onnxplugin build op_nn_onnx_plugin.so"
  echo "    --opapi build opapi_nn.so"
  echo "    --opgraph build opgraph_nn.so"
  echo "    --ophost build ophost_nn.so"
  echo "    --opkernel build binary kernel"
  echo "    --opkernel_aicpu build aicpu kernel"
  echo "    --opkernel_aicpu_test build and run aicpu opkernel unit tests"
  echo "    --pkg build run pkg"
  echo "    --jit build run pkg without kernel bin"
  echo "    --experimental Build experimental version"
  echo "    --run_example Compile and execute the example. use --run_example --help for more detail"
  echo "    --genop Create the initial directory for op, like: --genop=op_class/op_name"
  echo "    --genop_aicpu Create the initial directory for AI CPU op, like: --genop_aicpu=op_class/op_name"
  echo "    --oom Build with oom mode on the kernel side, with options: '-g --cce-enable-oom'"
  echo "    --dump_cce Dump kernel precompiled files (.i) for debugging"
  echo "    --bisheng_flags Specify bisheng compiler config, like: --bisheng_flags=ccec_g,oom, use ',' to separate different compiler flags"
  echo "    --kernel_template_input Specify kernel template input arguments, like: --kernel_template_input="args0=args0;args1=args1;args2=args2;args3=args3""
  echo "                                                                                                  Use ';' to separate different kernel template args, can only specify a single kernel template input"
  echo "to be continued ..."
}

# 检查--help 前的组合参数是否非法
check_help_combinations() {
  local args=("$@")
  local has_u=false
  local has_test_command=false
  local has_build_command=false
  local has_pkg=false
  local has_opkernel=false
  local has_opkernel_aicpu=false

  for arg in "${args[@]}"; do
    case "$arg" in
      -u) has_u=true ;;
      --ophost | --opapi | --onnxplugin | --opgraph)
        has_test_command=true
        has_build_command=true
        ;;
      --pkg) has_pkg=true ;;
      --opkernel) has_opkernel=true ;;
      --opkernel_aicpu) has_opkernel_aicpu=true ;;
      --help | -h) ;;
    esac
  done

  # 检查help中的无效命令组合
  if [[ "$has_pkg" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    print_error "--pkg cannot be used with test(-u, etc.), --ophost, --opapi, --opgraph"
    return 1
  fi

  if [[ "$has_opkernel" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    print_error "--opkernel cannot be used with test(-u, etc.), --ophost, --opapi, --opgraph"
    return 1
  fi

  if [[ "$has_opkernel_aicpu" == "true" && ("$has_test_command" == "true" || "$has_u" == "true") ]]; then
    echo "[ERROR] --opkernel_aicpu cannot be used with test(-u, --ophost, etc.), --ophost, --opapi, or --opgraph"
    return 1
  fi

  return 0
}

check_param() {
  # --ops不能与--ophost，--opapi, --opgraph同时存在，如果带U则可以
  if [[ -n "$COMPILED_OPS" && "$ENABLE_TEST" == "FALSE" ]] && [[ "$OP_HOST" == "TRUE" || "$OP_GRAPH" == "TRUE" || "$OP_API" == "TRUE" ]]; then
    print_error "--ops cannot be used with --ophost, --opapi"
    exit 1
  fi

  # --pkg不能与-u（UT模式，包含_test的参数）或者--ophost，--opapi, --opgraph同时存在
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    if [[ "$ENABLE_TEST" == "TRUE" ]]; then
      print_error "--pkg cannot be used with test(-u, --ophost, etc.)"
      exit 1
    fi

    if [[ "$OP_HOST" == "TRUE" || "$OP_GRAPH" == "TRUE" || "$OP_API" == "TRUE" ]]; then
      print_error "--pkg cannot be used with --ophost, --opapi, --opgraph"
      exit 1
    fi
  fi

  if [[ -n "${BUILD_TYPE}" ]]; then
    if [[ "${BUILD_TYPE}" != "Release" && "${BUILD_TYPE}" != "Debug" ]]; then
      echo "[ERROR] --build-type only support Release/Debug Mode"
      exit 1
    fi
  fi

  if [[ "${BUILD_TYPE}" == "Debug" ]]; then
    if [[ "$ENABLE_MSSANITIZER" == "TRUE" || "$ENABLE_OOM" == "TRUE" || "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
      echo "[ERROR] --build-type=Debug cannot be used with --mssanitizer, --oom, --dump_cce"
      exit 1
    fi
  fi

  if [ -n "$BISHENG_FLAGS" ]; then
    if [[ "$ENABLE_MSSANITIZER" == "TRUE" || "$ENABLE_OOM" == "TRUE" || "$ENABLE_DUMP_CCE" == "TRUE" ]]; then
      echo "[ERROR] --bisheng_flags= cannot be used with --mssanitizer, --oom, --dump_cce"
      exit 1
    fi
  fi

  if [[ "$ENABLE_MSSANITIZER" == "TRUE" && "$ENABLE_OOM" == "TRUE" ]]; then
    echo "[ERROR] --mssanitizer cannot be used with --oom"
    exit 1
  fi

  if $(echo ${USE_CMD} | grep -wq "opkernel") && $(echo ${USE_CMD} | grep -wq "jit"); then
    print_error "--opkernel cannot be used with --jit"
    exit 1
  fi

  if $(echo ${USE_CMD} | grep -wq "opkernel_aicpu") && $(echo ${USE_CMD} | grep -wq "jit"); then
    echo "[ERROR] --opkernel_aicpu cannot be used with --jit"
    exit 1
  fi

  if [ -n "$KERNEL_TEMPLATE_INPUT" ]; then
    if [[ -z "${COMPILED_OPS}" || "$COMPILED_OPS" == *","* ]]; then
      echo "[ERROR] --kernel_template_input must be used with --ops= and can only specify a single operator"
      exit 1
    fi
  fi
}

COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"
print_success() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_GREEN}[SUCCESS] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

set_create_libs() {
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    return
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" && "$ENABLE_CUSTOM" != "TRUE" ]]; then
    BUILD_LIBS=("ophost_${REPOSITORY_NAME}" "opapi_${REPOSITORY_NAME}" "op_${REPOSITORY_NAME}_onnx_plugin" "opgraph_${REPOSITORY_NAME}")
    ENABLE_CREATE_LIB=TRUE
  else
    if [[ "$OP_HOST" == "TRUE" ]]; then
      BUILD_LIBS+=("ophost_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_API" == "TRUE" ]]; then
      BUILD_LIBS+=("opapi_${REPOSITORY_NAME}")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$ONNX_PLUGIN" == "TRUE" ]]; then
      BUILD_LIBS+=("op_${REPOSITORY_NAME}_onnx_plugin")
      ENABLE_CREATE_LIB=TRUE
    fi
    if [[ "$OP_GRAPH" == "TRUE" ]]; then
 	       BUILD_LIBS+=("opgraph_${REPOSITORY_NAME}")
 	       ENABLE_CREATE_LIB=TRUE
 	  fi
  fi
}

set_ut_mode() {
  if [[ "$ENABLE_TEST" != "TRUE" ]]; then
    return
  fi
  ENABLE_CUSTOM=FALSE
  UT_TEST_ALL=TRUE
  if [[ "$OP_HOST" == "TRUE" ]]; then
    OP_HOST_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_GRAPH" == "TRUE" ]]; then
 	     OP_GRAPH_UT=TRUE
 	     UT_TEST_ALL=FALSE
 	fi
  if [[ "$OP_API" == "TRUE" ]]; then
    OP_API_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL" == "TRUE" ]]; then
    OP_KERNEL_UT=TRUE
    UT_TEST_ALL=FALSE
  fi
  if [[ "$OP_KERNEL_AICPU" == "TRUE" ]]; then
    OP_KERNEL_AICPU_UT=TRUE
    UT_TEST_ALL=FALSE
  fi

  if [[ "$OP_KERNEL_AICPU_UT" == "TRUE" ]]; then
    UT_TEST_ALL=FALSE
  fi

  # 检查测试项，至少有一个
  if [[ "$UT_TEST_ALL" == "FALSE" && "$OP_HOST_UT" == "FALSE" && "$OP_GRAPH_UT" == "FALSE" && "$OP_API_UT" == "FALSE" && "$OP_KERNEL_UT" == "FALSE" && "$OP_KERNEL_AICPU_UT" == "FALSE" ]]; then
    print_error "At least one test target must be specified (ophost test, opapi test, opgraph test, opkernel test, opkernel_aicpu_test)"
    usage
    exit 1
  fi

  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_HOST_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_host_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_GRAPH_UT" == "TRUE" ]]; then
 	     UT_TARGES+=("${REPOSITORY_NAME}_op_graph_ut")
 	fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_API_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_api_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_op_kernel_ut")
  fi
  if [[ "$UT_TEST_ALL" == "TRUE" ]] || [[ "$OP_KERNEL_AICPU_UT" == "TRUE" ]]; then
    UT_TARGES+=("${REPOSITORY_NAME}_aicpu_op_kernel_ut")
  fi
}

make_clean() {
  if [ -d "$BUILD_PATH" ]; then
    cd $BUILD_PATH
    make clean
    print_success "make clean success!"
  fi
}

make_clean_all() {
  if [ -d "$BUILD_PATH" ]; then
    cd $BUILD_PATH
    rm -rf ./*
  fi
  [ -d "$BUILD_OUT_PATH" ] && rm -rf $BUILD_OUT_PATH
  print_success "make clean all success!"
}

checkopts() {
  THREAD_NUM=$(grep -c ^processor /proc/cpuinfo)
  VERBOSE=""
  BUILD_MODE=""
  COMPILED_OPS=""
  UT_TEST_ALL=FALSE
  CHANGED_FILES=""
  CI_MODE=FALSE
  COMPUTE_UNIT=""
  VENDOR_NAME=""
  SHOW_HELP=""
  OP_NAME=""
  EXAMPLE_NAME=""
  EXAMPLE_MODE=""
  USE_CMD="$*"
  KERNEL_TEMPLATE_INPUT=""

  BUILD_TYPE="Release"
  ENABLE_MSSANITIZER=FALSE
  ENABLE_OOM=FALSE
  ENABLE_DUMP_CCE=FALSE
  ENABLE_COVERAGE=FALSE
  ENABLE_UT_EXEC=TRUE
  ENABLE_ASAN=FALSE
  ENABLE_VALGRIND=FALSE
  ENABLE_BINARY=FALSE
  ENABLE_CUSTOM=FALSE
  ENABLE_STATIC=FALSE
  ENABLE_PACKAGE=FALSE
  ENABLE_JIT=FALSE
  ENABLE_TEST=FALSE
  ENABLE_EXPERIMENTAL=FALSE
  NO_FORCE=FALSE
  AICPU_ONLY=FALSE
  NO_AICPU=FALSE
  OP_API_UT=FALSE
  OP_HOST_UT=FALSE
  OP_GRAPH_UT=FALSE
  ONNX_PLUGIN=FALSE
  OP_KERNEL_UT=FALSE
  OP_KERNEL_AICPU_UT=FALSE
  OP_API=FALSE
  OP_HOST=FALSE
  OP_GRAPH=FALSE
  OP_KERNEL=FALSE
  OP_KERNEL_AICPU=FALSE
  ENABLE_CREATE_LIB=FALSE
  ENABLE_RUN_EXAMPLE=FALSE
  ENABLE_SIMULATOR=FALSE
  BISHENG_FLAGS=""
  BUILD_LIBS=()
  UT_TARGES=()

  ENABLE_GENOP=FALSE
  ENABLE_GENOP_AICPU=FALSE
  GENOP_TYPE=""
  GENOP_NAME=""
  GENOP_BASE=${BASE_PATH}
  MODULE_EXT=""
  NO_ACLNN=FALSE

  if [ $# -eq 0 ]; then
    usage "$SHOW_HELP"
    exit 0
  fi

  # 首先检查所有参数是否合法
  for arg in "$@"; do
    if [[ "$arg" =~ ^- ]]; then # 只检查以-开头的参数
      if ! check_option_validity "$arg"; then
        usage
        exit 1
      fi
    fi
  done

  # 检查并处理--help
  for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" ]]; then
      # 检查帮助信息中的组合参数
      check_help_combinations "$@"
      local comb_result=$?
      if [ $comb_result -eq 1 ]; then
        exit 1
      fi
      SHOW_HELP="general"

      # 检查 --help 前面的命令
      for prev_arg in "$@"; do
        case "$prev_arg" in
          --pkg) SHOW_HELP="pkg" ;;
          --opkernel) SHOW_HELP="opkernel" ;;
          --opkernel_aicpu) SHOW_HELP="opkernel_aicpu" ;;
          -u) SHOW_HELP="test" ;;
          --make_clean_all | --make_clean) SHOW_HELP="clean" ;;
          --ophost) SHOW_HELP="ophost" ;;
          --opgraph) SHOW_HELP="opgraph" ;;
          --opapi) SHOW_HELP="opapi" ;;
          --onnxplugin) SHOW_HELP="onnxplugin" ;;
          --run_example) SHOW_HELP="run_example" ;;
          --genop) SHOW_HELP="genop" ;;
          --genop_aicpu) SHOW_HELP="genop_aicpu" ;;
        esac
      done

      usage "$SHOW_HELP"
      exit 0
    fi
  done

  process_genop() {
    local opt_name=$1
    local genop_value=$2

    if [[ "$opt_name" == "genop" ]]; then
      ENABLE_GENOP=TRUE
    elif [[ "$opt_name" == "genop_aicpu" ]]; then
      ENABLE_GENOP_AICPU=TRUE
    else
      usage "genop"
      exit 1
    fi

    if [[ "$genop_value" != *"/"* ]] || [[ "$genop_value" == *"/" ]]; then
      usage "$opt_name"
      exit 1
    fi

    GENOP_NAME=${genop_value##*/}
    local remaining=${genop_value%/*}

    if [[ "$remaining" != *"/"* ]]; then
      GENOP_TYPE=$remaining
      GENOP_BASE=${BASE_PATH}
    else
      GENOP_TYPE=${remaining##*/}
      GENOP_BASE=${remaining%/*}
      if [[ ! "$GENOP_BASE" =~ ^/ && ! "$GENOP_BASE" =~ ^[a-zA-Z]: ]]; then
        GENOP_BASE="${BASE_PATH}/${GENOP_BASE}"
      fi
    fi
  }

  # Process the options
  while getopts $SUPPORTED_SHORT_OPTS opt; do
    case "${opt}" in
      h)
        usage
        exit 0
        ;;
      j) THREAD_NUM=$OPTARG ;;
      v) VERBOSE="VERBOSE=1" ;;
      O) BUILD_MODE="-O$OPTARG" ;;
      u) ENABLE_TEST=TRUE ;;
      f)
        CHANGED_FILES=$OPTARG
        CI_MODE=TRUE
        ;;
      -) case $OPTARG in
        help)
          usage
          exit 0
          ;;
        ops=*)
          COMPILED_OPS=${OPTARG#*=}
          ENABLE_CUSTOM=TRUE
          ;;
        genop=*)
          process_genop "genop" "${OPTARG#*=}"
          ;;
        genop_aicpu=*)
          process_genop "genop_aicpu" "${OPTARG#*=}"
          ;;
        soc=*)
          COMPUTE_UNIT=${OPTARG#*=}
          COMPUTE_UNIT=$(echo "$COMPUTE_UNIT" | sed 's/ascend950/ascend950/g')
          ;;
        vendor_name=*)
          VENDOR_NAME=${OPTARG#*=}
          ENABLE_CUSTOM=TRUE
          ;;
        cann_3rd_lib_path=*)
          CANN_3RD_LIB_PATH="$(realpath ${OPTARG#*=})"
          ;;
        example_name=*) EXAMPLE_NAME=${OPTARG#*=} ;;
        build-type=*)
          BUILD_TYPE=${OPTARG#*=}
          ;;
        module_extension=*)
          MODULE_EXT=${OPTARG#*=}
          ;;
        noaclnn) NO_ACLNN=TRUE ;;
        mssanitizer) ENABLE_MSSANITIZER=FALSE ;;
        oom) ENABLE_OOM=TRUE ;;
        dump_cce) ENABLE_DUMP_CCE=TRUE ;;
        noexec) ENABLE_UT_EXEC=FALSE ;;
        noaicpu) NO_AICPU=TRUE ;;
        cov) ENABLE_COVERAGE=TRUE;;
        opkernel)
          ENABLE_BINARY=TRUE
          OP_KERNEL=TRUE
          ;;
        opkernel_aicpu)
          OP_KERNEL_AICPU=TRUE
          ;;
        opkernel_aicpu_test)
          OP_KERNEL_AICPU_UT=TRUE
          ENABLE_TEST=TRUE
          ;;
        static)
          ENABLE_STATIC=TRUE
          ENABLE_BINARY=TRUE
          ;;
        pkg)
          ENABLE_PACKAGE=TRUE
          ENABLE_BINARY=TRUE
          ;;
        jit)
          ENABLE_JIT=TRUE
          ;;
        asan) ENABLE_ASAN=TRUE ;;
        run_example) ENABLE_RUN_EXAMPLE=TRUE
          set_example_opt "$@"
          ;;
        experimental) ENABLE_EXPERIMENTAL=TRUE ;;
        make_clean_all) make_clean_all
                        exit 0 ;;
        make_clean) make_clean
                    exit 0 ;;
        no_force) NO_FORCE=TRUE ;;
        simulator) ENABLE_SIMULATOR=TRUE ;;
        bisheng_flags=*)
          BISHENG_FLAGS=${OPTARG#*=}
          ;;
        kernel_template_input=*)
          KERNEL_TEMPLATE_INPUT=${OPTARG#*=}
          ;;
        *)
          ## 如果不在RELEASE_TARGETS，不做处理
          if ! in_array "$OPTARG" "${RELEASE_TARGETS[@]}"; then
            print_error "Invalid option: --$OPTARG"
            usage
            exit 1
          fi

          if [[ "$OPTARG" == "ophost" ]]; then
            OP_HOST=TRUE
          elif [[ "$OPTARG" == "opgraph" ]]; then
 	          OP_GRAPH=TRUE
          elif [[ "$OPTARG" == "opapi" ]]; then
            OP_API=TRUE
          elif [[ "$OPTARG" == "opkernel" ]]; then
            OP_KERNEL=TRUE
          elif [[ "$OPTARG" == "opkernel_aicpu" ]]; then
            OP_KERNEL_AICPU=TRUE
          elif [[ "$OPTARG" == "onnxplugin" ]]; then
            ONNX_PLUGIN=TRUE
          else
            usage
            exit 1
          fi
          ;;
      esac ;;
      *)
        print_error "Undefined option: ${opt}"
        usage
        exit 1
        ;;
    esac
  done

  if [[ "$OP_KERNEL_AICPU_UT" != "TRUE" && "$ENABLE_TEST" == "TRUE" && "$OP_HOST" == "FALSE" && "$OP_GRAPH" == "FALSE" && "$OP_API" == "FALSE" && "$OP_KERNEL" == "FALSE" ]]; then
    OP_HOST=TRUE
    OP_GRAPH=TRUE
    OP_API=TRUE
    OP_KERNEL=TRUE
  fi
  if [[ "$ENABLE_JIT" == "TRUE" ]]; then
    ENABLE_PACKAGE=TRUE
    ENABLE_BINARY=FALSE
  fi
  check_param
  set_create_libs
  set_ut_mode
}

set_example_opt() {
  if [[ $OPTIND -le $# && ${!OPTIND} != -* ]]; then
    OP_NAME=${!OPTIND}
    ((OPTIND++))
  fi
  if [[ $OPTIND -le $# && ${!OPTIND} != -* ]]; then
    EXAMPLE_MODE=${!OPTIND}
    ((OPTIND++))
  fi
  if [[ $OPTIND -le $# && ${!OPTIND} != -* ]]; then
    PKG_MODE=${!OPTIND}
    ((OPTIND++))
  fi
}

custom_cmake_args() {
  if [[ -n $COMPILED_OPS ]]; then
    COMPILED_OPS="${COMPILED_OPS//,/;}"
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_OP_NAME=${COMPILED_OPS}"
  else
    CMAKE_ARGS="$CMAKE_ARGS -UASCEND_OP_NAME"
  fi
  if [[ -n $VENDOR_NAME ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DVENDOR_NAME=${VENDOR_NAME}"
  else
    CMAKE_ARGS="$CMAKE_ARGS -UVENDOR_NAME"
  fi
}

assemble_cmake_args() {
  CMAKE_ARGS="-DBUILD_PATH=${BUILD_PATH}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_VALGRIND=${ENABLE_VALGRIND}"
  CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_BUILD_TYPE=${BUILD_TYPE}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_MSSANITIZER=${ENABLE_MSSANITIZER}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_OOM=${ENABLE_OOM}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_DUMP_CCE=${ENABLE_DUMP_CCE}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_COVERAGE=${ENABLE_COVERAGE}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_TEST=${ENABLE_TEST}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_UT_EXEC=${ENABLE_UT_EXEC}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_CUSTOM=${ENABLE_CUSTOM}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_STATIC=${ENABLE_STATIC}"
  CMAKE_ARGS="$CMAKE_ARGS -DMODULE_EXT=${MODULE_EXT}"
  if [[ "$NO_ACLNN" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DNO_ACLNN=TRUE"
  fi
  custom_cmake_args
  if [[ "$ENABLE_ASAN" == "TRUE" ]]; then
    set +e
    echo 'int main() {return 0;}' | gcc -x c -fsanitize=address - -o asan_test >/dev/null 2>&1
    if [ $? -ne 0 ]; then
      echo "This environment does not have the ASAN library, no need enable ASAN"
      ENABLE_ASAN=FALSE
    else
      $(rm -f asan_test)
      CMAKE_ARGS="$CMAKE_ARGS -DENABLE_ASAN=TRUE"
    fi
    set -e
  fi
  if [[ "$ENABLE_CUSTOM" == "TRUE" ]]; then
    ENABLE_BINARY=TRUE
  fi
  if [[ "$NO_AICPU" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DNO_AICPU=TRUE"
  fi
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_BINARY=${ENABLE_BINARY}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PACKAGE=${ENABLE_PACKAGE}"
  CMAKE_ARGS="$CMAKE_ARGS -DENABLE_EXPERIMENTAL=${ENABLE_EXPERIMENTAL}"
  CMAKE_ARGS="$CMAKE_ARGS -DNO_FORCE=${NO_FORCE}"
  CMAKE_ARGS="$CMAKE_ARGS -DBUILD_MODE=${BUILD_MODE}"
  CMAKE_ARGS="$CMAKE_ARGS -DOP_HOST_UT=${OP_HOST_UT}"
  CMAKE_ARGS="$CMAKE_ARGS -DOP_GRAPH_UT=${OP_GRAPH_UT}"
  CMAKE_ARGS="$CMAKE_ARGS -DOP_API_UT=${OP_API_UT}"
  CMAKE_ARGS="$CMAKE_ARGS -DOP_KERNEL_UT=${OP_KERNEL_UT}"
  CMAKE_ARGS="$CMAKE_ARGS -DOP_KERNEL_AICPU_UT=${OP_KERNEL_AICPU_UT}"
  CMAKE_ARGS="$CMAKE_ARGS -DUT_TEST_ALL=${UT_TEST_ALL}"
  if [[ "x$BISHENG_FLAGS" != "x" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DBISHENG_FLAGS=${BISHENG_FLAGS}"
  fi
  if [[ -n $COMPUTE_UNIT ]]; then
    IFS=',' read -ra COMPUTE_UNIT <<<"$COMPUTE_UNIT"
    COMPUTE_UNIT_SHORT=""
    for unit in "${COMPUTE_UNIT[@]}"; do
      for support_unit in "${SUPPORT_COMPUTE_UNIT_SHORT[@]}"; do
        lowercase_word=$(echo "$unit" | tr '[:upper:]' '[:lower:]')
        if [[ "$lowercase_word" == *"$support_unit"* ]]; then
          COMPUTE_UNIT_SHORT="$COMPUTE_UNIT_SHORT$support_unit;"
          break
        fi
      done
    done

    if [[ -z $COMPUTE_UNIT_SHORT ]]; then
      print_error "The soc [${COMPUTE_UNIT}] is not support."
      usage
      exit 1
    fi

    echo "COMPUTE_UNIT: ${COMPUTE_UNIT_SHORT}"
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_COMPUTE_UNIT=$COMPUTE_UNIT_SHORT"
  else
    CMAKE_ARGS="$CMAKE_ARGS -UASCEND_COMPUTE_UNIT"
  fi
  CMAKE_ARGS="$CMAKE_ARGS -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
  if [[ "x$KERNEL_TEMPLATE_INPUT" != "x" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DKERNEL_TEMPLATE_INPUT=${KERNEL_TEMPLATE_INPUT}"
    NO_FORCE=TRUE
    CMAKE_ARGS="$CMAKE_ARGS -DNO_FORCE=${NO_FORCE}"
  fi
}

build_static_lib() {
  echo $dotted_line
  echo "Start to build static lib."

  git submodule init && git submodule update
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..
  local all_targets=$(cmake --build . --target help)
  rm -fr ${BUILD_PATH}/bin_tmp
  mkdir -p ${BUILD_PATH}/bin_tmp
  if grep -wq "ophost_nn_static" <<< "${all_targets}"; then
    cmake --build . --target ophost_nn_static -- ${VERBOSE} -j $THREAD_NUM
  fi

  local UNITS=(${COMPUTE_UNIT_SHORT//;/ })
  if [[ ${#UNITS[@]} -eq 0 ]]; then
    UNITS+=("ascend910b")
  fi
  cmake --build . --target opapi_nn_static -- ${VERBOSE} -j $THREAD_NUM
  local jit_command=""
  if [[ "$ENABLE_JIT" == "TRUE" ]]; then
    jit_command="-j"
  fi
  for unit in "${UNITS[@]}"; do
    rm -fr ${BUILD_PATH}/autogen/${unit}
    mkdir -p ${BUILD_PATH}/bin_tmp/${unit}
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" GenStaticOpResourceIni -s ${unit} -b ${BUILD_PATH} ${jit_command}
    python3 "${BASE_PATH}/scripts/util/build_opp_kernel_static.py" StaticCompile -s ${unit} -b ${BUILD_PATH} -n=0 -a=${ARCH_INFO} ${jit_command}
  done
  cmake --build . --target cann_nn_static -- ${VERBOSE} -j $THREAD_NUM
  print_success "Build static lib success!"
}

build_lib() {
  echo $dotted_line
  echo "Start to build libs ${BUILD_LIBS[@]}"

  git submodule init && git submodule update
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} -UENABLE_STATIC ..
  local all_targets=$(cmake --build . --target help)
  for lib in "${BUILD_LIBS[@]}"; do
    if grep -wq "$lib" <<< "${all_targets}"; then
      echo "Building target ${lib}"
      cmake --build . --target ${lib} -- ${VERBOSE} -j $THREAD_NUM
    else
      echo "No need to build target ${lib}"
    fi
  done

  print_success "Build libs ${BUILD_LIBS[@]} success!"
}

build_binary() {
  echo $dotted_line
  echo "Start to build binary"

  mkdir -p "${BUILD_PATH}"
  echo "--------------- build tiling start ---------------"
  local tilingLib="ophost_${REPOSITORY_NAME}"
  local tilingSo="${BUILD_PATH}/lib${tilingLib}.so"
  rm -f ${tilingSo}  # 临时方案，简单实现，强制先删再编，tiling代码不变时仅浪费链接时间
  local TMP_BUILD_LIBS=${BUILD_LIBS}
  local TMP_CMAKE_ARGS=${CMAKE_ARGS}
  CMAKE_ARGS="-DBUILD_PATH=$BUILD_PATH -DENABLE_CUSTOM=FALSE -DENABLE_PACKAGE=FALSE"
  BUILD_LIBS+=("${tilingLib}")
  build_lib
  BUILD_LIBS=${TMP_BUILD_LIBS}
  CMAKE_ARGS=${TMP_CMAKE_ARGS}

  export ASCEND_OPP_PATH=${BUILD_PATH}/opp
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${BUILD_PATH}/common/stub/op_tiling
  local fakeTilingSoPath=${BUILD_PATH}/opp/op_impl/built-in/ai_core/tbe/op_tiling/lib/linux/$(arch)
  mkdir -p ${fakeTilingSoPath}
  ln -sf ${tilingSo} ${fakeTilingSoPath}/liboptiling.so
  ln -sf ${tilingSo} ${fakeTilingSoPath}/libopmaster_ct.so
  rm -f ${BUILD_PATH}/opp/scene.info
  echo "os=linux"     >> ${BUILD_PATH}/opp/scene.info
  echo "os_version="  >> ${BUILD_PATH}/opp/scene.info
  echo "arch=$(arch)" >> ${BUILD_PATH}/opp/scene.info
  echo "--------------- build tiling end ---------------"

  cd "${BUILD_PATH}" && cmake .. ${CMAKE_ARGS}

  echo "--------------- prepare build start ---------------"
  local all_targets=$(cmake --build . --target help)
  if grep -wq "ascendc_impl_gen" <<< "${all_targets}"; then
    cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi
  if grep -wq "gen_bin_scripts" <<< "${all_targets}"; then
    cmake --build . --target gen_bin_scripts -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi
  echo "--------------- prepare build end ---------------"

  echo "--------------- binary build start ---------------"
  local UNITS=(${COMPUTE_UNIT_SHORT//;/ })
  if [[ ${#UNITS[@]} -eq 0 ]]; then
    UNITS+=("ascend910b")
  fi
  for unit in "${UNITS[@]}"; do
    remove_opc_cmd="rm -rf ${BUILD_PATH}/binary/${unit}/bin/opc_cmd"
    ${remove_opc_cmd}
    if grep -wq "prepare_binary_compile_${unit}" <<< "${all_targets}"; then
      cmake --build . --target prepare_binary_compile_${unit} -- ${VERBOSE} -j 1
      if [ $? -ne 0 ]; then
        print_error "opc gen failed!" && exit 1
      fi
    fi
    OPC_CMD_FILE="${BUILD_PATH}/binary/${unit}/bin/opc_cmd/opc_cmd.sh"
    [[ -f "$OPC_CMD_FILE" ]] && opc_list_num=$(wc -l < "$OPC_CMD_FILE") || opc_list_num=0
    CMAKE_ARGS="${CMAKE_ARGS} -DOPC_NUM_${unit}=${opc_list_num}"
  done
  cd "$BUILD_PATH" && cmake .. ${CMAKE_ARGS}

  if grep -wq "binary" <<< "${all_targets}"; then
    cmake --build . --target binary -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then
      print_error "Kernel compile failed!" && exit 1
    fi
  fi
  if grep -wq "gen_bin_info_config" <<< "${all_targets}"; then
    cmake --build . --target gen_bin_info_config -- ${VERBOSE} -j $THREAD_NUM
    if [ $? -ne 0 ]; then exit 1; fi
  fi

  print_success "Build binary success!"
}

build_pkg() {
  echo "--------------- build pkg start ---------------"
  local all_targets=$(cmake --build . --target help)
  if [[ "$ENABLE_BINARY" == "FALSE" ]]; then # for jit need dynamic py
    if grep -wq "ascendc_impl_gen" <<< "${all_targets}"; then
      cmake --build . --target ascendc_impl_gen -- ${VERBOSE} -j $THREAD_NUM
      if [ $? -ne 0 ]; then exit 1; fi
    fi
  fi
  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} ..

  if echo "${all_targets}" | grep -wq "build_es_nn"; then
 	     cmake --build . --target build_es_nn -- ${VERBOSE} -j $THREAD_NUM
 	     [ $? -ne 0 ] && echo "[ERROR] target:build_es_nn compile failed!" && exit 1
 	fi
  cmake --build . --target package -- ${VERBOSE} -j $THREAD_NUM

  print_success "Build package success!"
}

parse_op_dependencies() {
  echo $dotted_line
  echo "Start to parse op dependencies for ${COMPILED_OPS}"

  cd "${BUILD_PATH}"
  python3 ${BASE_PATH}/scripts/util/dependency_parser.py --ops ${COMPILED_OPS} -p ${BUILD_PATH}
  echo $dotted_line
  echo "End to parse op dependencies"
}

set_ci_mode() {
  cd ${BASE_PATH}
  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    {
      result=$(python3 ${BASE_PATH}/scripts/util/parse_changed_files.py ${CHANGED_FILES} ${ENABLE_EXPERIMENTAL})
    } || {
      echo $result && exit 1
    }
    for line in $result; do
      if [[ $line == op_* ]]; then
        TRIGGER_UTS+=($line)
      fi
    done
  else
    {
      result=$(python3 ${BASE_PATH}/scripts/util/parse_compile_changed_files.py ${CHANGED_FILES} ${ENABLE_EXPERIMENTAL})
    } || {
      echo $result && exit 1
    }
    args=(${result//:/ })
    CMAKE_ARGS="$CMAKE_ARGS -DASCEND_OP_NAME=${args[0]} -DASCEND_COMPILE_OPS=${args[1]}"
  fi
}

build_ut() {
  echo $dotted_line
  echo "Start to build ut"

  git submodule init && git submodule update
  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi
  # 删除ai_core下的json文件，强制UT执行时重新生成json文件，避免多次执行之间的干扰
  cd "${BUILD_PATH}"  && rm -rf ${BUILD_PATH}/tbe/op_info_cfg/ai_core/* && cmake ${CMAKE_ARGS} ..
  local enable_cov=FALSE
  if [[ "$CI_MODE" == "TRUE" ]]; then
    # ci 模式
    for trigger_option in "${TRIGGER_UTS[@]}"; do
 	    ut_args=(${trigger_option//:/ })
      if [[ "${UT_TARGES[@]}" =~ "${ut_args[0]}" ]]; then
        enable_cov=TRUE
        echo "Trigger Ut: ${ut_args[0]} for ops: ${ut_args[1]}"
 	        if [[ ${ut_args[3]} == "default" ]]; then
 	          cmake ${CMAKE_ARGS} -DASCEND_OP_NAME=${ut_args[1]} -DASCEND_COMPILE_OPS=${ut_args[2]} ..
 	        else
 	          cmake ${CMAKE_ARGS} -DASCEND_OP_NAME=${ut_args[1]} -DASCEND_COMPILE_OPS=${ut_args[2]} -DASCEND_COMPUTE_UNIT=${ut_args[3]} ..
 	        fi
        cmake --build . --target ${REPOSITORY_NAME}_${ut_args[0]} -- ${VERBOSE} -j $THREAD_NUM
      else
        echo "Not need trigger Ut: ${ut_args[0]}"
      fi
    done
  else
    enable_cov=TRUE
    if echo "${UT_TARGES[@]}" | grep -v "aicpu_op_kernel" | grep -q "op_kernel"; then
      local all_targets=$(cmake --build . --target help)
      if grep -wq "pre_op_kernel_ut" <<< "${all_targets}"; then
        echo "exec pre_op_kernel_ut..."
        cmake --build . --target pre_op_kernel_ut -- ${VERBOSE} -j $THREAD_NUM
        cmake ${CMAKE_ARGS} ..
      fi
    fi
    cmake --build . --target ${UT_TARGES[@]} -- ${VERBOSE} -j $THREAD_NUM
  fi

  if [[ "$ENABLE_COVERAGE" =~ "TRUE" && "$enable_cov" == "TRUE" ]]; then
    cmake --build . --target generate_ops_cpp_cov -- ${VERBOSE} -j $THREAD_NUM
  fi
}

build_single_example() {
  echo "Start to run example,op_name:${OP_NAME} example_name:${example} mode:${EXAMPLE_MODE}."

  if [[ "${ENABLE_SIMULATOR}" == "TRUE" ]]; then
    if [[ "${EXAMPLE_MODE}" == "graph" ]]; then
      usage "run_example"
      exit 1
    fi
    # 根据soc设置仿真库路径
    if [[ -n $COMPUTE_UNIT_SHORT ]]; then
      IFS=';' read -ra COMPUTE_UNITS <<<"$COMPUTE_UNIT_SHORT"
      unit=${COMPUTE_UNITS[0]}
    else
      unit="ascend910b"
    fi
    if [[ "${SOC_TO_ARCH[${unit}]}x" == "x" ]]; then
      usage "run_example"
      exit 1
    fi
    SIMULATOR_PATH="${ASCEND_HOME_PATH}/${ARCH_INFO}-linux/simulator/dav_${SOC_TO_ARCH[${unit}]}/lib"
    if [[ ! -f ${SIMULATOR_PATH}/libruntime_camodel.so ]]; then
      echo "[ERROR] ${SIMULATOR_PATH}/libruntime_camodel.so not found."
      exit 1
    fi
    if [[ ! -f ${SIMULATOR_PATH}/libnpu_drv_camodel.so ]]; then
      echo "[ERROR] ${SIMULATOR_PATH}/libnpu_drv_camodel.so not found."
      exit 1
    fi
    rm -fr ${BUILD_PATH}/simulator
    mkdir -p ${BUILD_PATH}/simulator
    ln -sf ${SIMULATOR_PATH}/libruntime_camodel.so ${BUILD_PATH}/simulator/libruntime.so
    ln -sf ${SIMULATOR_PATH}/libnpu_drv_camodel.so ${BUILD_PATH}/simulator/libascend_hal.so
    echo "[INFO] Successfully linked simulator libraries: ${SIMULATOR_PATH}/libruntime_camodel.so, ${SIMULATOR_PATH}/libnpu_drv_camodel.so"
    export LD_LIBRARY_PATH=${BUILD_PATH}/simulator:${SIMULATOR_PATH}:${LD_LIBRARY_PATH}
  fi

  if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
    if [[ "${PKG_MODE}" == "cust" ]]; then
      if [[ "${VENDOR_NAME}" == "" ]]; then
        VENDOR_NAME="custom"
      fi
      export CUST_LIBRARY_PATH="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}_nn/op_api/lib"     # 仅自定义算子需要
      export CUST_INCLUDE_PATH="${ASCEND_HOME_PATH}/opp/vendors/${VENDOR_NAME}_nn/op_api/include" # 仅自定义算子需要
      export LD_LIBRARY_PATH=${CUST_LIBRARY_PATH}:${LD_LIBRARY_PATH}
      if [ -f ${EAGER_LIBRARY_PATH}/libascendcl.so ]; then
        g++ ${file} -I ${CUST_INCLUDE_PATH} -I ${INCLUDE_PATH} -I ${INCLUDE_PATH}/aclnnop -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} -lcust_opapi -lopapi_math -lascendcl -lnnopbase -o test_aclnn_${example} -Wl,-rpath=${CUST_LIBRARY_PATH}
      else
        g++ ${file} -I ${CUST_INCLUDE_PATH} -I ${INCLUDE_PATH} -I ${INCLUDE_PATH}/aclnnop -L ${CUST_LIBRARY_PATH} -L ${EAGER_LIBRARY_PATH} -lcust_opapi -lopapi_math -lacl_rt -lnnopbase -o test_aclnn_${example} -Wl,-rpath=${CUST_LIBRARY_PATH}
      fi
    elif [[ "${PKG_MODE}" == "" ]]; then
      if [ -f ${EAGER_LIBRARY_PATH}/libascendcl.so ] || [ -f ${EAGER_LIBRARY_OPP_PATH}/libascendcl.so]; then
        g++ ${file} -I ${INCLUDE_PATH} -I ${INCLUDE_PATH}/aclnnop -I ${ACLNN_INCLUDE_PATH} -L ${EAGER_LIBRARY_OPP_PATH} -L ${EAGER_LIBRARY_PATH} -lopapi_nn -lopapi_math -lascendcl -lnnopbase -o test_aclnn_${example}
      else
        g++ ${file} -I ${INCLUDE_PATH} -I ${INCLUDE_PATH}/aclnnop -I ${ACLNN_INCLUDE_PATH} -L ${EAGER_LIBRARY_OPP_PATH} -L ${EAGER_LIBRARY_PATH} -lopapi_nn -lopapi_math -lacl_rt -lnnopbase -o test_aclnn_${example}
      fi
    else
      usage "run_example"
      exit 1
    fi
  elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
    g++ ${file} -I ${GRAPH_INCLUDE_PATH} -I ${GE_INCLUDE_PATH} -I ${INCLUDE_PATH} -I ${EXTERNAL_INCLUDE_PATH} -I ${LINUX_INCLUDE_PATH} -I ${INC_INCLUDE_PATH} \
                -L ${GRAPH_LIBRARY_STUB_PATH} -L ${GRAPH_LIBRARY_PATH} \
                -lgraph -lge_runner -lgraph_base -lge_compiler -o test_geir_${example}
  fi
  ${BUILD_PATH}/"${pattern}${example}" && success_example+=(${example}) && echo -e "\n$dotted_line\nRun ${pattern}${example} success.\n$dotted_line\n"
}

build_example() {
  echo $dotted_line

  if [ ! -d "${BUILD_PATH}" ]; then
    mkdir -p "${BUILD_PATH}"
  fi
  cd "${BUILD_PATH}"
  if [[ "${EXAMPLE_MODE}" == "eager" ]]; then
    pattern="test_aclnn_"
  elif [[ "${EXAMPLE_MODE}" == "graph" ]]; then
    pattern="test_geir_"
  else
    usage "run_example"
    exit 1
  fi

  local grep_word="-v"

  if [[ "${ENABLE_EXPERIMENTAL}" == "TRUE" ]]; then
    grep_word=""
  fi

  OLDIFS=$IFS
  IFS=$'\n'
  {
    files=($(find ../ -path "*/${OP_NAME}/examples/${pattern}*.cpp" -not -path "*/opgen/template/*" | grep ${grep_word} "experimental"))
  } || {
    files=()
    echo "INFO: not find ${OP_NAME} A2/A3 examples."
  }
  if [[ "$COMPUTE_UNIT" == "ascend950" ]]; then
    files=($(find ../ -path "*/${OP_NAME}/examples/arch35/${pattern}*.cpp" | grep ${grep_word} "experimental"))
  fi
  if [[ "$COMPUTE_UNIT" == "ascend310p" ]]; then
    files=($(find ../ -path "*/${OP_NAME}/examples/arch20/${pattern}*.cpp" | grep ${grep_word} "experimental"))
  fi
  IFS=$OLDIFS

  failed_example=()
  success_example=()
  examples=()
  for file in "${files[@]}"; do
    example=${file#*"${pattern}"}
    example=${example%.*}
    examples+=($example)
    local old_ld_path=${LD_LIBRARY_PATH}
    if [[ $EXAMPLE_NAME == "" || $EXAMPLE_NAME == $example ]]; then
      build_single_example || {
        echo -e "\n$dotted_line\nRun ${pattern}${example} failed. \n$dotted_line\n"
        failed_example+=(${example})
      }
    fi
    export LD_LIBRARY_PATH=${old_ld_path}
  done

  examples=$(IFS=,; echo "${examples[*]}")
  failed_example=$(IFS=,; echo "${failed_example[*]}")
  success_example=$(IFS=,; echo "${success_example[*]}")

  if [[ ${#files[@]} -eq 0 ]]; then
    echo "ERROR: ${OP_NAME} doesn't have ${EXAMPLE_MODE} examples."
  elif [[ ${failed_example} != "" || ${success_example} == "" ]]; then
    echo "Run ${EXAMPLE_MODE} example failed, op_name: ${OP_NAME}, \
success examples: ${success_example}, failed_example: ${failed_example[@]}."
    exit 1
  fi
}

gen_op() {
  if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    print_error "op_class or op_name is not set."
    usage "genop"
    exit 1
  fi

  echo $dotted_line
  echo "Start to create the initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  # 检查 python 或 python3 是否存在
  local python_cmd=""
  if command -v python3 &>/dev/null; then
    python_cmd="python3"
  elif command -v python &>/dev/null; then
    python_cmd="python"
  fi

  if [ -n "${python_cmd}" ]; then
    ${python_cmd} "${BASE_PATH}/scripts/opgen/opgen_standalone.py" -t ${GENOP_TYPE} -n ${GENOP_NAME} -p ${GENOP_BASE}
    return $?
  fi
}

gen_aicpu_op() {
if [[ -z "$GENOP_NAME" ]] || [[ -z "$GENOP_TYPE" ]]; then
    print_error "op_class or op_name is not set."
    usage "genop"
    exit 1
  fi

  echo $dotted_line
  echo "Start to create the AI CPU initial directory for ${GENOP_NAME} under ${GENOP_TYPE}"

  # 检查 python 或 python3 是否存在
  local python_cmd=""
  if command -v python3 &> /dev/null; then
      python_cmd="python3"
  elif command -v python &> /dev/null; then
      python_cmd="python"
  fi

  if [ -n "${python_cmd}" ]; then
    ${python_cmd} "${BASE_PATH}/scripts/opgen/opgen_standalone.py" -t ${GENOP_TYPE} -n ${GENOP_NAME} -p ${GENOP_BASE} -v aicpu
    echo "Create the AI CPU initial directory for ${GENOP_NAME} under ${GENOP_TYPE} success"
    return $?
  else
    echo "Please install Python to generate op project framework."
  fi
}

package_static() {
    # Check weather BUILD_OUT_PATH directory exists
    if [ ! -d "$BUILD_OUT_PATH" ]; then
        echo "Error: Directory $BUILD_OUT_PATH does not exist."
        return 1
    fi

    # Check weather *.run is exists and verify the file numbers
    local run_files=("$BUILD_OUT_PATH"/*.run)
    if [ ${#run_files[@]} -eq 0 ]; then
        echo "Error: No .run files found in $BUILD_OUT_PATH directory."
        return 1
    fi
    if [ ${#run_files[@]} -gt 1 ]; then
        echo "Error: Multiple .run files found in $BUILD_OUT_PATH directory."
        printf '%s\n' "${run_files[@]}"
        return 1
    fi

    # Get filename of *.run file and set new directory name
    local run_file=$(basename "${run_files[0]}")
    echo "Found .run file: $run_file"
    if [[ "$run_file" != *"ops-nn"* ]]; then
        echo "Error: Filename '$run_file' does not contain 'ops-nn'."
        return 1
    fi
    local new_name="${run_file/ops-nn/ops-nn-static}"
    new_name="${new_name%.run}"

    # Check weather $BUILD_PATH/static_library_files directory exists and not empty
    local static_files_dir="$BUILD_PATH/static_library_files"
    if [ ! -d "$static_files_dir" ]; then
        echo "Error: Directory $static_files_dir does not exist."
        return 1
    fi
    if [ -z "$(ls -A "$static_files_dir")" ]; then
        echo "Error: Directory $static_files_dir is empty."
        return 1
    fi

    # Rename directory
    local new_dir_path="$BUILD_PATH/$new_name"
    if mv "$static_files_dir" "$new_dir_path"; then
        echo "Preparing for packaging: renamed $static_files_dir to $new_dir_path"
    else
        echo "Packaging preparation failed: directory rename failed ($static_files_dir -> $new_dir_path)"
        return 1
    fi

    # Create compressed package and restore directory name
    local new_filename="${new_name}.tar.gz"
    if tar -czf "$BUILD_OUT_PATH/$new_filename" -C "$BUILD_PATH" "$new_name"; then
        echo "Successfully created compressed package: $BUILD_OUT_PATH/$new_filename"
        # Restore original directory name
        echo "Restoring original directory name: $new_dir_path -> $static_files_dir"
        mv "$new_dir_path" "$static_files_dir"
        return 0
    else
        echo "Error: Failed to create compressed package."
        # Attempt to restore original directory name
        mv "$new_dir_path" "$static_files_dir"
        return 1
    fi
}


main() {
  checkopts "$@"
  if [[ "$ENABLE_GENOP" == "TRUE" ]]; then
    gen_op
    exit $?
  fi
  if [[ "$ENABLE_GENOP_AICPU" == "TRUE" ]]; then
    gen_aicpu_op
    exit $?
  fi
  assemble_cmake_args
  echo "CMAKE_ARGS: ${CMAKE_ARGS}"

  if [[ "$ENABLE_RUN_EXAMPLE" == "TRUE" ]]; then
    build_example
    exit $?
  fi
  mkdir -p "${BUILD_PATH}"
  cd "${BUILD_PATH}" && rm -f CMakeCache.txt && cmake -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH} -DENABLE_EXPERIMENTAL=${ENABLE_EXPERIMENTAL} -DPREPROCESS_ONLY=ON ..

  if [[ "$CI_MODE" == "TRUE" ]]; then
    set_ci_mode
  elif [[ -n $COMPILED_OPS ]]; then
    parse_op_dependencies
  fi

  cd "${BUILD_PATH}" && cmake ${CMAKE_ARGS} -DPREPROCESS_ONLY=OFF ..

  if [[ "$ENABLE_TEST" == "TRUE" ]]; then
    build_ut
    exit $?
  fi

  if [ "$ENABLE_CREATE_LIB" == "TRUE" ]; then
    build_lib
  fi
  if [[ "$ENABLE_BINARY" == "TRUE" || "$ENABLE_CUSTOM" == "TRUE" ]] && [[ "$ENABLE_JIT" == "FALSE" ]]; then
    build_binary
  fi
  if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
    build_static_lib
  fi
  if [[ "$ENABLE_PACKAGE" == "TRUE" || "$ENABLE_JIT" == "TRUE" ]]; then
    build_pkg
    if [[ "$ENABLE_STATIC" == "TRUE" ]]; then
      package_static
    fi
  fi
}

set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
