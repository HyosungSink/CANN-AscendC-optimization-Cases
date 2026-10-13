# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_FLOYD_OpProtoSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention/op_host/fused_floyd_attention_infershape.cpp
)

set(_FLOYD_OpTilingSourcesExt)
file(GLOB _Src1 "${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention/op_host/fused_floyd_attention_tiling*.cpp")
list(APPEND _FLOYD_OpTilingSourcesExt ${_Src1})


set(_FLOYD_OpKernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention/op_host/fused_floyd_attention_tiling.h
)

set(_FLOYD_OpKernelSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention/op_kernel/fused_floyd_attention.cpp
)

set(_FLOYD_OpKernelPrivateIncludesExt
        ${OPS_TRANSFORMER_DIR}/common/include/kernel
)

set(_FLOYD_OpKernelPrivateCompileDefinitionsExt
        KernelCtrlParam fused_floyd_attention fp16 ORIG_DTYPE_QUERY=DT_FLOAT16 DTYPE_DQ=half       KFC_L1_RESERVER_SIZE=0
        KernelCtrlParam fused_floyd_attention bf16 ORIG_DTYPE_QUERY=DT_BF16    DTYPE_DQ=bfloat16_t KFC_L1_RESERVER_SIZE=0
)


OpsTest_Level2_AddOp(
        SUB_SYSTEM                                transformer
        BRIEF                                     Floyd
        SNAKE                                     fused_floyd_attention
        PROTO_SOURCES_EXT                         ${_FLOYD_OpProtoSourcesExt}
        TILING_SOURCES_EXT                        ${_FLOYD_OpTilingSourcesExt}
        KERNEL_SOURCES_EXT                        ${_FLOYD_OpKernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                  ${_FLOYD_OpKernelTilingDataDefH}
        KERNEL_PRIVATE_INCLUDES_EXT               ${_FLOYD_OpKernelPrivateIncludesExt}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT    ${_FLOYD_OpKernelPrivateCompileDefinitionsExt}
)
