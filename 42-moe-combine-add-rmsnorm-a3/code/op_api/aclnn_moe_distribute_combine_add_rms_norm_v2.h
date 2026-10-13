/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef OP_API_INC_MOE_DISTRIBUTE_ADD_RMS_NORM_COMBINE
#define OP_API_INC_MOE_DISTRIBUTE_ADD_RMS_NORM_COMBINE

#include <string>

#include "aclnn/aclnn_base.h"
#include "aclnn_util.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 算子功能：实现aclnnMoeDistributeCombineAddRmsNorm功能。
 * @brief aclnnMoeDistributeCombineAddRmsNorm的第一段接口，根据具体的计算流程，计算workspace大小。
 * @domain aclnn_ops_infer
 * @param [in] expandX: 计算输入，Tensor，数据类型bfloat16，必须为2维，数据格式支持ND。
 * @param [in] expertIds: 计算输入，Tensor，数据类型int32，必须为2维，数据格式支持ND。
 * @param [in] assistInfoForCombine: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND，传输给combine算子的辅助信息。
 * @param [in] epSendCounts: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。
 * @param [in] expertScales: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND。
 * @param [in] residualX: 计算输入，Tensor，数据类型bfloat16，必须为3维，数据格式支持ND。
 * @param [in] gamma: 计算输入，Tensor，数据类型bfloat16，必须为1维，数据格式支持ND。
 * @param [in] tpSendCountsOptional: 计算输入，Tensor，数据类型int32，必须为1维，数据格式支持ND。无tp域通信时传空。
 * @param [in] xActiveMaskOptional: 计算输入，Tensor，数据类型bool，必须为1维，数据格式支持ND。预留参数，暂未使用，传空即可。
 * @param [in] activationScaleOptional: 计算输入，Tensor，数据类型float32，必须为1维，数据格式支持ND, GMM外抛的左矩阵量化系数，
    当x的类型为int32时，该参数必须有。预留参数，暂未使用，传空即可。
 * @param [in] weightScaleOptional: 计算输入，Tensor，数据类型float32，必须为2维，数据格式支持ND, GMM外抛的右矩阵量化系数，
    当x的类型为int32时，该参数必须有。预留参数，暂未使用，传空即可。
 * @param [in] groupListOptional: 计算输入，Tensor，数据类型int64，必须为1维，数据格式支持ND, GMM的分组大小，
    当weight scale的E>1时，该参数必须有。预留参数，暂未使用，传空即可。
 * @param [in] expandScalesOptional: 计算输入，Tensor，数据类型float32，必须为1维，数据格式支持ND。在昇腾910_93中暂未使用，传空即可。
 * @param [in] sharedExpertXOptional: 计算可选输入，Tensor, 数据类型float16，bfloat16，必须为2维，数据格式支持ND。数据类型必须与expandX类型保持一致，
    当sharedExpertRankNum不为0时，该参数必须为空。
 * @param [in] elasticInfoOptional: 计算可选输入，Tensor，数据类型int32，必须为1维，数据格式支持ND，支持非连续的Tensor。
 * @param [in] oriXOptional: 计算可选输入，Tensor，数据类型bfloat16，必须为2维，数据格式支持ND，支持非连续的Tensor。
    当copyExpertNum不为0或constExpertNum不为0时必须传入有效输入。
 * @param [in] constExpertAlpha1Optional: 计算可选输入，Tensor，数据类型float16，bfloat16，int32，必须为1维，数据格式支持ND，支持非连续的Tensor。数据类型必须与expandX保持一致。
    当constExpertNum不为0时必须传入有效输入。
 * @param [in] constExpertAlpha2Optional: 计算可选输入，Tensor，数据类型float16，bfloat16，int32，必须为1维，数据格式支持ND，支持非连续的Tensor。数据类型必须与expandX保持一致。
    当constExpertNum不为0时必须传入有效输入。
 * @param [in] constExpertVOptional: 计算可选输入，Tensor，数据类型float16，bfloat16，int32，必须为1维，数据格式支持ND，支持非连续的Tensor。数据类型必须与expandX保持一致。 
 * @param [in] groupEp: 计算输入，str。ep通信域名称，专家并行的通信域。字符串长度范围为[1, 128)，不能和groupTp相同。
 * @param [in] epWorldSize: 计算输入，int。ep通信域size。在昇腾910_93场景中取值支持8/16/32/64/128/144/256/288。
 * @param [in] epRankId: 计算输入，int。ep本卡Id。取值范围[0, epWorldSize)，同一个EP通信域中各卡的epRankId不能重复。
 * @param [in] moeExpertNum: 计算输入，int。MOE专家数量。取值范围[1, 512]，且需
    满足moeExpertNum%(epWorldSize-sharedExpertRankNum)等于0。
 * @param [in] groupTp: 计算可选输入，str。tp通信域名称，数据并行的通信域。无tp通信域时传空，
    有tp通信域时字符串长度范围为[1, 128)，不能和groupEp相同。
 * @param [in] tpWorldSize: 计算可选输入，int。tp通信域size。取值范围[0, 2]，0和1表示无tp域通信，有tp域通信时仅支持2。
 * @param [in] tpRankId: 计算可选输入，int。tp本卡Id。取值范围[0, tpWorldSize)，同一个TP通信域中各卡的tpRankId不能重复。
 * @param [in] expertShardType: 计算可选输入，int。专家共享类型。当前仅支持传0。
 * @param [in] sharedExpertNum: 计算可选输入，int。共享专家数量。取值范围[0,1]。0表示无共享专家。
 * @param [in] sharedExpertRankNum: 计算可选输入，int。共享专家数量。支持传0表示无共享专家卡，不为0时需满足
    sharedExpertRankNum<epWorldSize且epWorldSize%sharedExpertRankNum等于0。
 * @param [in] globalBs: 计算可选输入，int。在910_93中，当每个rank的Bs数一致场景下，globalBs = Bs * epWorldSize 或
    globalBs = 0；当每个rank的Bs数不一致场景下，globalBs = maxBs * epWorldSize，其中maxBs表示单卡Bs最大值。
 * @param [in] outDtype: 计算可选输入，int。输出数据类型。预留参数，暂未使用，传0即可。
 * @param [in] commQuantMode: 计算可选输入，int。通信量化类型。预留参数，暂未使用，传0即可。
 * @param [in] groupListType: 计算可选输入，int。groupList格式。预留参数，暂未使用，传0即可。
 * @param [in] commAlg: 计算可选输入，str。通信算法类型。预留参数，暂未使用。
 * @param [in] normEps: 计算可选输入，float。用于防止AddRmsNorm除0错误，数据类型仅支持float，默认值为1e-6。
 * @param [in] zeroExpertNum: 计算输入，int64_t。取值范围[0,MAX_INT32]，合法的零专家ID值是[moeExpertNum,moeExpertNum+zeroExpertNum)。
 * @param [in] copyExpertNum: 计算输入，int64_t。取值范围[0,MAX_INT32]，合法的零专家ID值是[moeExpertNum+zeroExpertNum,moeExpertNum+zeroExpertNum+copyExpertNum)。
 * @param [in] constExpertNum: 计算输入，int64_t。取值范围[0,MAX_INT32]，合法的零专家ID值是[moeExpertNum+zeroExpertNum+copyExpertNum,moeExpertNum+zeroExpertNum+copyExpertNum+constExpertNum)。
 * @param [out] yOut: 计算输出，Tensor，必选输出，数据类型支持bfloat16，仅支持3维，数据格式支持ND。
 * @param [out] rstdOut: 计算输出，Tensor，必选输出，数据类型支持float32，仅支持3维，数据格式支持ND。
 * @param [out] xOut: 计算输出，Tensor，必选输出，数据类型支持bfloat16，仅支持3维，数据格式支持ND。
 * @param [out] workspaceSize: 出参，返回需要在npu device侧申请的workspace大小。
 * @param [out] executor: 出参，返回op执行器，包含了算子计算流程。
 * @return aclnnStatus: 返回值，返回状态码
 *
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2GetWorkspaceSize(const aclTensor* expandX, const aclTensor* expertIds,
                                                                            const aclTensor* assistInfoForCombine, const aclTensor* epSendCounts,
                                                                            const aclTensor* expertScales, const aclTensor* residualX,
                                                                            const aclTensor* gamma, const aclTensor* tpSendCountsOptional,
                                                                            const aclTensor* xActiveMaskOptional, const aclTensor* activationScaleOptional,
                                                                            const aclTensor* weightScaleOptional, const aclTensor* groupListOptional,
                                                                            const aclTensor* expandScalesOptional,  const aclTensor* sharedExpertXOptional,
                                                                            const aclTensor* elasticInfoOptional, const aclTensor* oriXOptional, 
                                                                            const aclTensor* constExpertAlpha1Optional, const aclTensor* constExpertAlpha2Optional, 
                                                                            const aclTensor* constExpertVOptional, const char* groupEp, int64_t epWorldSize, 
                                                                            int64_t epRankId, int64_t moeExpertNum, const char* groupTp, int64_t tpWorldSize, 
                                                                            int64_t tpRankId, int64_t expertShardType, int64_t sharedExpertNum, 
                                                                            int64_t sharedExpertRankNum, int64_t globalBs, int64_t outDtype,
                                                                            int64_t commQuantMode, int64_t groupListType, const char* commAlg, float normEps,
                                                                            int64_t zeroExpertNum, int64_t copyExpertNum, int64_t constExpertNum,
                                                                            aclTensor* yOut, aclTensor* rstdOut, aclTensor* xOut, uint64_t* workspaceSize,
                                                                            aclOpExecutor** executor);

/**
 * @brief aclnnMoeDistributeCombineAddRmsNorm的第二段接口，用于执行计算。
 * @param [in] workspace: 在npu device侧申请的workspace内存起址。
 * @param [in] workspace_size: 在npu device侧申请的workspace大小，由第一段接口aclnnMoeDistributeCombineAddRmsNormGetWorkspaceSize获取。
 * @param [in] executor: op执行器，包含了算子计算流程。
 * @param [in] stream: acl stream流。
 * @return aclnnStatus: 返回状态码
 */
ACLNN_API aclnnStatus aclnnMoeDistributeCombineAddRmsNormV2(void* workspace, uint64_t workspaceSize, aclOpExecutor* executor,
                                                          aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif  // OP_API_INC_MOE_DISTRIBUTE_ADD_RMS_NORM_COMBINE