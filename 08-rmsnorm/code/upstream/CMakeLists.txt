# ----------------------------------------------------------------------------
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)
set(PKG_NAME nn)
project(${PKG_NAME} VERSION 1.0.0)

set(CMAKE_C_FLAGS_RELEASE "")
set(CMAKE_CXX_FLAGS_RELEASE "")

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
endif()

if(PROJECT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  message(STATUS "compile project with library")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG
         "Build ops-nn with cann pkg" ON)
else()
  message(STATUS "compile project with src")
  option(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG
         "Build ops-nn with cann source" OFF)
endif()

if(UNIX)
  set(SYSTEM_PREFIX ${CMAKE_SYSTEM_PROCESSOR}-linux)
endif()
set(OPS_NN_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# 外部传参
option(ENABLE_ASAN "Enable asan" OFF)
option(ENABLE_VALGRIND "Enable valgrind" OFF)
option(ENABLE_TEST "Enable test" OFF)
option(ENABLE_UT_EXEC "Enable exec ut" OFF)
option(ENABLE_BINARY "Enable build binary" OFF)
option(ENABLE_CUSTOM "Enable build custom" OFF)
option(ENABLE_PACKAGE "Enable build package" OFF)
option(ENABLE_EXPERIMENTAL "Enable experimental module" OFF)
option(ENABLE_COVERAGE "Enable coverage" OFF)
option(ENABLE_MSSANITIZER "Enable mssanitizer mode" OFF)
option(ENABLE_OOM "Enable oom mode" OFF)
option(ENABLE_DUMP_CCE "Enable dump_cce mode" OFF)
option(NO_AICPU "Disable aicpu" OFF)
option(OP_HOST_UT "Enable ophost ut" OFF)
option(OP_API_UT "Enable opapi ut" OFF)
option(OP_GRAPH_UT "Enable graph ut" OFF)
option(OP_KERNEL_UT "Enable kernel ut" OFF)
option(OP_KERNEL_AICPU_UT "Enable aicpu kernel ut" OFF)
option(UT_TEST_ALL "Enable all ut" OFF)

set(BISHENG_FLAGS "" CACHE STRING "bisheng compiler flags")
set(CANN_3RD_LIB_PATH ${PROJECT_SOURCE_DIR}/build/third_party CACHE STRING "cann third party lib path")

if(ENABLE_TEST)
    set(DEFAULT_BUILD_TYPE "Debug")
else()
    set(DEFAULT_BUILD_TYPE "Release")
endif()

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "${DEFAULT_BUILD_TYPE}" CACHE STRING "Choose the build type: Release/Debug" FORCE)
endif()

set(BUILD_MODE
    ""
    CACHE STRING "build mode -O0/O1/O2")
set(ASCEND_COMPUTE_UNIT
    "ascend910b"
    CACHE STRING "soc that need to be compiled")
set(ASCEND_ALL_COMPUTE_UNIT
    "ascend310p;ascend910;ascend910b;ascend910_93;ascend950;ascend031;ascend035;ascend310b;ascend910_55;mc62cm12a;kirinx90;kirin9030"
    CACHE STRING "all soc list")
set(VENDOR_NAME
    "custom"
    CACHE STRING "vendor name")
set(KERNEL_TEMPLATE_INPUT "" CACHE STRING "kernel template input")

if(NOT ${BUILD_MODE} STREQUAL "" AND NOT ${BUILD_MODE} STREQUAL "FALSE")
  set(COMPILE_OP_MODE ${BUILD_MODE})
else()
  if(ENABLE_TEST)
    set(COMPILE_OP_MODE "-O0")
  else()
    set(COMPILE_OP_MODE -O2)
  endif()
endif()

# Cmake Compile配置
set(CMAKE_CXX_STANDARD
    17
    CACHE STRING "c++17 is needed for this project")
include(cmake/third_party/opbase.cmake) # 放在dependencies.cmake前面
include(cmake/dependencies.cmake)
include(cmake/third_party/nlohmann_json.cmake)
include(cmake/third_party/abseil-cpp.cmake)
include(cmake/third_party/ascend_protobuf.cmake)
include(cmake/variables.cmake)
include(cmake/func.cmake)
include(cmake/opbuild.cmake)
include(cmake/third_party/eigen.cmake)
include(cmake/intf_pub_linux.cmake)
include(cmake/modules/Finddlog.cmake)
include(version.cmake)

if(ENABLE_CUSTOM AND ENABLE_ASC_BUILD)
  include(cmake/custom_kernel.cmake)
endif()

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  check_pkg_build_deps("ops-nn")
endif()
add_version_info_targets()

if(ENABLE_TEST)
  # ut common dependencies
  include(cmake/intf_pub_llt_gccnative.cmake)
  include(cmake/ut.cmake)
  include(cmake/third_party/gtest.cmake)

  # op host ut
  if(UT_TEST_ALL OR OP_HOST_UT)
    set(OP_TILING_MODULE_NAME ${PKG_NAME}_op_tiling_ut)

    set(OP_INFERSHAPE_MODULE_NAME ${PKG_NAME}_op_infershape_ut)
  endif()

  # op api ut
  if(UT_TEST_ALL OR OP_API_UT)
    set(OP_API_MODULE_NAME ${PKG_NAME}_op_api_ut)
  endif()

  # op kernel ut
  if(UT_TEST_ALL OR OP_KERNEL_UT)
    set(UT_DONE OFF)
    include(cmake/gen_ops_info.cmake)
    set(OP_KERNEL_MODULE_NAME ${PKG_NAME}_op_kernel_ut)
  endif()
  add_subdirectory(tests/ut/common)
endif()

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  add_subdirectory(common)
endif()

add_category_subdirectory()

if(ENABLE_TEST)
  if(UT_TEST_ALL OR OP_GRAPH_UT)
    include(cmake/gen_ops_info.cmake)
 	  include(cmake/symbol.cmake)
    set(OP_GRAPH_MODULE_NAME ${PKG_NAME}_op_graph_ut)
  endif()
  add_subdirectory(tests/ut)
endif()

message(STATUS "Ops for this compilation contains: ${COMPILED_OPS}")

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND NOT ENABLE_TEST)
  include(cmake/gen_ops_info.cmake)
  gen_ops_info_and_python()

  include(cmake/symbol.cmake)
  if(ENABLE_CUSTOM)
    # custom package library setting and install
    gen_cust_symbol()
  else()
    gen_norm_symbol()
  endif()
  if(ENABLE_STATIC)
    include(cmake/static.cmake)
  endif()
endif()
if(NOT BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG AND TARGET ${OPHOST_NAME}_opapi_obj)
  add_dependencies(${OPHOST_NAME}_opapi_obj opbuild_gen_inner)
  target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE ${CMAKE_BINARY_DIR}/asl/ops/cann/ops/built-in/op_host/inner)
endif()

if(ENABLE_PACKAGE)
  include(cmake/package.cmake)

  if(ENABLE_CUSTOM)
    pack_custom()
  else()
    pack_built_in()
  endif()
endif()

install(DIRECTORY ${OPS_CMCT}/
        DESTINATION ${IMPL_INSTALL_DIR}/common/cmct/
)