# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        rotary_position_embedding_def.cpp
    )
    add_modules_sources(OPTYPE rotary_position_embedding ACLNNTYPE aclnn_exclude)
endif()

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    set(rotary_position_embedding_apt_depends posembedding/apply_rotary_pos_emb CACHE INTERNAL "Dependencies for rotary_position_embedding")
    add_ops_compile_options(
        OP_NAME RotaryPositionEmbedding
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
            OP_NAME RotaryPositionEmbedding
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
    )
else()
    add_modules_sources(OPTYPE rotary_position_embedding ACLNNTYPE aclnn_exclude)
endif()