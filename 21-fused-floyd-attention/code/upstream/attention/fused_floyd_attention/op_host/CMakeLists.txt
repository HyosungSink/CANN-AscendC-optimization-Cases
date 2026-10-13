# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        fused_floyd_attention_def.cpp
    )
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME FusedFloydAttention
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
        )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME FusedFloydAttention
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
    set(fused_floyd_attention_depends attention/common CACHE INTERNAL "Dependencies for fused_floyd_attention")
else()
    add_ops_compile_options(
            OP_NAME FusedFloydAttention
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
    )
    add_modules_sources(OPTYPE fused_floyd_attention ACLNNTYPE aclnn)
endif()
