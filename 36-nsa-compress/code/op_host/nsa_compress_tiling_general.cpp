/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file nsa_compress_tiling_general.cpp
 * \brief
 */

#include <cmath>
#include "nsa_compress_tiling.h"
#include "nsa_compress_tiling_common.h"
#include "../common/include/err/ops_err.h"
using namespace Ops::Transformer::OpTiling;

namespace optiling {
namespace Nsa {
class NsaCompressTiling : public TilingBaseClass {
public:
    explicit NsaCompressTiling(gert::TilingContext *context) : TilingBaseClass(context) {};
    ~NsaCompressTiling() override = default;

protected:
    ge::graphStatus CheckContext()
    {
        auto attrs = context_->GetAttrs();
        OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);
        // 除了第一个属性都是必选属性
        size_t idx = 1;
        auto compressBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto compressStridePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto actSeqLenTypePtr = attrs->GetAttrPointer<int64_t>(idx++);
        size_t *workspaces = context_->GetWorkspaceSizes(1);

        OP_CHECK_NULL_WITH_CONTEXT(context_, compressBlockSizePtr);
        OP_CHECK_NULL_WITH_CONTEXT(context_, compressStridePtr);
        OP_CHECK_NULL_WITH_CONTEXT(context_, actSeqLenTypePtr);
        OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);

        // 检查input、weight和outputCache
        auto inputShape = context_->GetInputShape(INPUT_INPUT_INDEX);
        auto weightShape = context_->GetInputShape(WEIGHT_INPUT_INDEX);

        OP_CHECK_NULL_WITH_CONTEXT(context_, inputShape);
        OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData());
        OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetRawTilingData()->GetData());
        OP_CHECK_IF(context_->GetRawTilingData()->GetCapacity() < tilingData.GetDataSize(),
                   OPS_REPORT_VECTOR_INNER_ERR(opName,
                                               "context tiling data capacity %zu < actual tiling data size %zu.",
                                               context_->GetRawTilingData()->GetCapacity(), tilingData.GetDataSize()),
                   return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    }

    bool AnalyzeInputShape()
    {
        auto inputShape = context_->GetInputShape(Zero);

        headNum = inputShape->GetStorageShape().GetDim(One);
        headDim = inputShape->GetStorageShape().GetDim(Two);
        headNumDim = headNum * headDim;
        totalKvSize = inputShape->GetStorageShape().GetDim(0) * headNumDim;
        return true;
    }

    bool AnalyzeAttrs()
    {
        auto attrs = context_->GetAttrs();

        size_t idx = 0;
        inputLayout = attrs->GetAttrPointer<char>(idx++);
        auto compressBlockSizePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto compressStridePtr = attrs->GetAttrPointer<int64_t>(idx++);
        auto actSeqLenTypePtr = attrs->GetAttrPointer<int64_t>(idx++);

        compressBlockSize = static_cast<uint32_t>(*compressBlockSizePtr);
        compressStride = static_cast<uint32_t>(*compressStridePtr);
        actSeqLenType = *actSeqLenTypePtr;

        weightSize = compressBlockSize * headNum;
        maxOverlap = 2u * (compressBlockSize + compressStride - 1u) / compressStride - 1u;
        return true;
    }

    bool AnalyzeDtype()
    {
        // 目前默认转换成float32计算
        inputDtype = context_->GetInputDesc(INPUT_INPUT_INDEX)->GetDataType();
        inputDtypeBytes = ge::GetSizeByDataType(inputDtype);
        calcTypeSize = ge::GetSizeByDataType(ge::DT_FLOAT);
        if (!inputDtypeBytes || !calcTypeSize) {
            return false;
        }
        ubBlockHalfNum = MIN_COPY_UINT_SIZE / inputDtypeBytes; // 当前输入的Dtype下(fp16/bf16),对齐32B所需的元素个数
        ubBlockFloatNum = MIN_COPY_UINT_SIZE / calcTypeSize;   // 当前计算的Dtype下(fp32),对齐32B所需的元素个数

        switch (inputDtype) {
            case ge::DT_FLOAT16:
                break;
            case ge::DT_BF16:
                break;
            default:
                OPS_REPORT_VECTOR_INNER_ERR(opName, "not support input dtype: %s for now",
                                            ge::TypeUtils::DataTypeToSerialString(inputDtype).c_str());
                return false;
        }
        return true;
    }

    bool AnalyzeOptionalInput()
    {
        auto actSeqLenTensor = context_->GetOptionalInputTensor(ACT_SEQ_LEN_INPUT_INDEX);
        if (actSeqLenTensor == nullptr) {
            OP_LOGW(context_, "actSeqLenTensor is null pointer");
            return false;
        }
        auto &actSeqLenShape = actSeqLenTensor->GetShape().GetStorageShape();
        if (actSeqLenShape.GetDimNum() != 1 || actSeqLenShape.GetDim(Zero) == 0) {
            OP_LOGW(context_, "actSeqLenShape is invalid %lu %ld", actSeqLenShape.GetDimNum(),
                      actSeqLenShape.GetDim(Zero));
            return false;
        }
        /* Get Data from tensor. */
        const int64_t *value = actSeqLenTensor->GetData<int64_t>();
        if (value == nullptr) {
            OP_LOGW(context_, "actSeqLenTensor data is null pointer");
            return false;
        }
        batchSize = actSeqLenShape.GetDim(Zero);
        uint32_t pre_seq_len = 0;

        totalOutSeqLen = 0u;
        for (uint32_t i = 0; i < batchSize; ++i) {
            uint32_t valueI = static_cast<uint32_t>(value[i]);
            actualSeqLen.push_back(valueI);
            uint32_t cur_seq_len = valueI - pre_seq_len;
            if (cur_seq_len >= compressBlockSize) {
                totalOutSeqLen += (cur_seq_len - compressBlockSize + compressStride) / compressStride;
            }

            actualOutputSeqLen.push_back(totalOutSeqLen);
            pre_seq_len += cur_seq_len;
        }
        totalCompressSize = totalOutSeqLen * headNumDim;
        return true;
    }

    bool IsCapable() override
    {
        // do some check
        return true;
    };

    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override
    {
        auto platformInfoPtr = context_->GetPlatformInfo();
        if (platformInfoPtr == nullptr) {
            auto compileInfoPtr = context_->GetCompileInfo<NsaCompressCompileInfo>();
            OP_CHECK_IF(compileInfoPtr == nullptr, OPS_REPORT_VECTOR_INNER_ERR(opName, "compileInfoPtr is null."),
                       return ge::GRAPH_FAILED);
            aivNum = compileInfoPtr->aivNum;
            ubSize = compileInfoPtr->ubSize;
        } else {
            auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
            aivNum = ascendcPlatform.GetCoreNumAiv();
            ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        }
        OP_LOGI(context_,
                "get platform from compileInfo.aivNum(%u)"
                " ubSize(%lu).", aivNum, ubSize);
        return ge::GRAPH_SUCCESS;
    };
    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override
    {
        opName = context_->GetNodeName();
        OP_LOGD(opName,
                       "TilingContext:"
                       " %s.",
                       GetTilingContextDebugStr().c_str());
        OP_CHECK_IF(CheckContext() != ge::GRAPH_SUCCESS, OPS_REPORT_VECTOR_INNER_ERR(opName, "invalid context."),
                   return ge::GRAPH_FAILED);
        OP_CHECK_IF(!AnalyzeInputShape() || !AnalyzeAttrs() || !AnalyzeDtype() || !AnalyzeOptionalInput(),
                   OPS_REPORT_VECTOR_INNER_ERR(opName, "fail to analyze context info."), return ge::GRAPH_FAILED);
        return ge::GRAPH_SUCCESS;
    };

    void GetDivisors(uint32_t n)
    {
        if (n == 0u) {
            return;
        }

        for (uint32_t i = 1; i <= std::sqrt(n); ++i) {
            if (n % i == 0u) {
                // 如果i是n的因子，则添加i和n/i
                divisorNums.push_back(i);
                if (i != n / i) { // 避免重复添加平方根
                    divisorNums.push_back(n / i);
                }
            }
        }
        std::sort(divisorNums.begin(), divisorNums.end());
    }

    uint32_t CeilDiv(uint32_t a, uint32_t b)
    {
        if (b == 0u) {
            return b;
        }
        return (a + b - 1u) / b;
    }

    bool GetSplitInfo()
    {
        // 单位kv所需ub大小
        uint32_t unitKvFp16Memory = headDim * inputDtypeBytes; // 需要补乘subSeqlen 与 maxHeadPerCore
        uint32_t unitKvFp32Memory = headDim * calcTypeSize;    // 需要补乘subSeqlen 与 maxHeadPerCore

        // weight总共所需ub大小
        uint32_t unitWeightPadFp16Memory =
            compressBlockSize * inputDtypeBytes; // 需要补乘ceil(maxHeadPerCore, ubBlockHalfNum) *ubBlockHalfNum
        uint32_t unitWeightPadFp32Memory =
            compressBlockSize * calcTypeSize; // 需要补乘ceil(maxHeadPerCore, ubBlockHalfNum) *ubBlockHalfNum
        uint32_t unitWeightBroadCastFp32Memory =
            compressBlockSize * ubBlockFloatNum *
            calcTypeSize; // 需要补乘ceil(maxHeadPerCore, ubBlockHalfNum) *ubBlockHalfNum
        uint32_t unitWeightFp32Memory = compressBlockSize * ubBlockFloatNum * calcTypeSize; // 需要补乘maxHeadPerCore

        uint32_t unitSubResultMemory =
            headDim * calcTypeSize; // 单位中间结果subresult所需的ub大小， 需要补乘subSeqlen 与 maxHeadPerCore
        uint32_t unitOverlapMemory =
            maxOverlap * headDim * calcTypeSize;                   // 单位overlap所需ub大小, 需要补乘maxHeadPerCore
        uint32_t unitCompressKvMemory = headDim * inputDtypeBytes; // 单位输出所需的ub大小, 需要补乘maxHeadPerCore

        // 递增，先从能不切开始
        for (auto i = divisorNums.begin(); i != divisorNums.end(); i++) {
            divHeadNum = *i;
            singleHeadPerCore = headNum / divHeadNum;
            resHeadNum = headNum - singleHeadPerCore * divHeadNum;
            if (resHeadNum) {
                maxHeadPerCore = singleHeadPerCore + 1;
            } else {
                maxHeadPerCore = singleHeadPerCore;
            }

            divSeqNum = aivNum / divHeadNum;
            singleSeqPerCore = totalOutSeqLen / divSeqNum;
            resSeqNum = totalOutSeqLen - singleSeqPerCore * divSeqNum;

            NeedUBMemory = CeilDiv(maxHeadPerCore, ubBlockHalfNum) * ubBlockHalfNum *
                               (unitWeightPadFp16Memory + unitWeightPadFp32Memory + unitWeightBroadCastFp32Memory) +
                           maxHeadPerCore * unitWeightFp32Memory;
            NeedUBMemory += maxHeadPerCore * (unitOverlapMemory + unitCompressKvMemory);
            if (NeedUBMemory > static_cast<uint32_t>(ubSize)) {
                continue;
            }

            maxCopyKVTokensNums = (static_cast<uint32_t>(ubSize) - NeedUBMemory) /
                                  (maxHeadPerCore * (unitKvFp16Memory + unitKvFp32Memory + unitSubResultMemory));
            if (maxCopyKVTokensNums == 0) {
                continue;
            } else {
                maxCopyKVTokensNums = std::min(maxCopyKVTokensNums, compressBlockSize);
                return true;
            }
        }
        return false;
    }

    void GetTilingDataInfo()
    {
        // 确定每个核对应的head索引
        for (uint32_t i = 0; i < divSeqNum; i++) {
            for (uint32_t j = 0; j < divHeadNum; j++) {
                // 给每个核对应的head索引赋值，除不尽的余数均匀分配给前面的核
                if (j <= resHeadNum && j > 0u) {
                    PerCoreHeadIdx[i * divHeadNum + j] =
                        PerCoreHeadIdx[i * divHeadNum + j - 1u] + singleHeadPerCore + 1u;
                } else if (j > 0u) {
                    PerCoreHeadIdx[i * divHeadNum + j] = PerCoreHeadIdx[i * divHeadNum + j - 1u] + singleHeadPerCore;
                }
                // 给每个核对应的输出output数量赋值，一组head的输出output数相同，除不尽的余数均匀分配给前面的核
                if (i < resSeqNum) {
                    perCoreOutputNumsArr[i * divHeadNum + j] = singleSeqPerCore + 1u;
                } else {
                    perCoreOutputNumsArr[i * divHeadNum + j] = singleSeqPerCore;
                }
                if (j == 0u && i < divSeqNum - 1u) {
                    perCoreStartOutputOffset[(i + 1u) * divHeadNum] =
                        perCoreStartOutputOffset[i * divHeadNum] + perCoreOutputNumsArr[i * divHeadNum] * headNumDim;
                } else if (j > 0u) {
                    perCoreStartOutputOffset[i * divHeadNum + j] =
                        perCoreStartOutputOffset[i * divHeadNum] + PerCoreHeadIdx[i * divHeadNum + j] * headDim;
                }
            }
        }

        for (uint32_t i = 0; i < aivNum - 1u; i++) {
            if (PerCoreHeadIdx[i + 1u]) {
                PerCoreHeadNum[i] = PerCoreHeadIdx[i + 1u] - PerCoreHeadIdx[i];
            } else {
                PerCoreHeadNum[i] = headNum - PerCoreHeadIdx[i];
            }
        }
        PerCoreHeadNum[aivNum - 1u] = headNum - PerCoreHeadIdx[aivNum - 1u];

        for (uint32_t i = 0; i < divSeqNum; i++) {
            startOutputToken = i > 0u ? startOutputToken + perCoreOutputNumsArr[(i - 1u) * divHeadNum] : 0u;
            while (startOutputToken >= actualOutputSeqLen[curSampleIdx]) {
                startTokens = actualSeqLen[curSampleIdx];
                curSampleIdx += 1u;
            }
            overOutputTokens =
                curSampleIdx > 0 ? startOutputToken - actualOutputSeqLen[curSampleIdx - 1] : startOutputToken;
            for (uint32_t j = 0; j < divHeadNum; j++) {
                kvStartBatchPerCore[i * divHeadNum + j] = curSampleIdx;
                kvStartTokenPerCore[i * divHeadNum + j] = overOutputTokens * compressStride + startTokens;
                kvStartOffsetPerCore[i * divHeadNum + j] =
                    kvStartTokenPerCore[i * divHeadNum] * headNumDim + PerCoreHeadIdx[j] * headDim;
            }
        }
    }
    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override
    {
        GetDivisors(aivNum); // 求解aivNum的因子
        OP_CHECK_IF(!GetSplitInfo(),
                OP_LOGE(context_->GetNodeName(),
                "Unable to process current input size and parameters."),
                return ge::GRAPH_FAILED);
        GetTilingDataInfo();

        tilingData.set_InputDtype(inputDtype);
        tilingData.set_TotalKvSize(totalKvSize);
        tilingData.set_WeightSize(weightSize);
        tilingData.set_BatchSize(batchSize);
        tilingData.set_TotalCompressSize(totalCompressSize);
        tilingData.set_CompressBlockSize(compressBlockSize);
        tilingData.set_CompressStride(compressStride);
        tilingData.set_MaxOverlap(maxOverlap);
        tilingData.set_HeadDim(headDim);
        tilingData.set_HeadNum(headNum);
        tilingData.set_DivHeadNums(divHeadNum);
        tilingData.set_DivSeqNums(divSeqNum);
        tilingData.set_BlocksNums(maxCopyKVTokensNums);
        tilingData.set_PerCoreOutputNum(perCoreOutputNumsArr);
        tilingData.set_PerCoreStartOutputOffset(perCoreStartOutputOffset);
        tilingData.set_PerCoreHeadIdx(PerCoreHeadIdx);
        tilingData.set_PerCoreHeadNum(PerCoreHeadNum);
        tilingData.set_kvStartBatchIdx(kvStartBatchPerCore);
        tilingData.set_kvStartTokenIdx(kvStartTokenPerCore);
        tilingData.set_kvStartOffset(kvStartOffsetPerCore);

        return ge::GRAPH_SUCCESS;
    };
    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override
    {
        return ge::GRAPH_SUCCESS;
    };
    // 5、计算TilingKey
    uint64_t GetTilingKey() const override
    {
        return 0;
    };
    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override
    {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
        uint32_t sysWorkSpaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
        size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
        currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkSpaceSize);
        return ge::GRAPH_SUCCESS;
    };
    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override
    {
        context_->SetBlockDim(aivNum);
        tilingData.SaveToBuffer(context_->GetRawTilingData()->GetData(), context_->GetRawTilingData()->GetCapacity());
        context_->GetRawTilingData()->SetDataSize(tilingData.GetDataSize());
        return ge::GRAPH_SUCCESS;
    };

    ge::DataType inputDtype;
    uint32_t inputDtypeBytes;
    uint32_t calcTypeSize;

    const char *opName = "NsaCompress";
    uint32_t aivNum;
    uint64_t ubSize;
    uint32_t NeedUBMemory = 0;

    const char *inputLayout;

    uint32_t batchSize;
    uint32_t compressBlockSize;
    uint32_t compressStride;
    uint32_t headNum;
    uint32_t headDim;
    uint32_t headNumDim;

    uint32_t totalKvSize;       // 输入的总元素数量
    uint32_t weightSize;        // weight的总元素数量
    uint32_t totalOutSeqLen;    // 输出的总seq_len数量
    uint32_t totalCompressSize; // 输出的总元素数量

    uint32_t maxOverlap;          // 当前compressBlockSize和compressStride下,中间结果overlap的数量
    uint32_t maxCopyKVTokensNums; // 计算搬运至UB块的kv tokens数量
    uint32_t actSeqLenType;

    uint32_t divSeqNum = 0;        // 总输出token的分多少core组
    uint32_t singleSeqPerCore = 0; // 每个core处理多少seq
    uint32_t resSeqNum = 0;        // 分seq后的余数

    uint32_t divHeadNum = 0;        // 每个core组分多少核
    uint32_t singleHeadPerCore = 0; // 每个core处理的head数
    uint32_t resHeadNum = 0;        // 分head后的余数
    uint32_t maxHeadPerCore = 0;    // 处理的最大Head

    uint32_t startOutputToken = 0;
    uint32_t curSampleIdx = 0;
    uint32_t startTokens = 0;
    uint32_t overOutputTokens = 0;

    uint32_t ubBlockHalfNum;  // 当前输入的Dtype下(fp16/bf16),对齐32B所需的元素个数
    uint32_t ubBlockFloatNum; // 当前计算的Dtype下(fp32),对齐32B所需的元素个数

    std::vector<uint32_t> actualSeqLen;                    // 各batch的seq_len长度，前缀和方式
    std::vector<uint32_t> actualOutputSeqLen;              // 各batch可以输出的token数，前缀和方式
    std::vector<uint32_t> divisorNums;                     // aivNum的因子
    uint32_t PerCoreHeadIdx[MAX_CORE_NUM] = {0};           // 记录每个核处理的head索引
    uint32_t PerCoreHeadNum[MAX_CORE_NUM] = {0};           // 记录每个核处理的head数量
    uint32_t kvStartTokenPerCore[MAX_CORE_NUM] = {0};      // 计算每个core所需搬运的首个kv块在输入中的seq索引数
    uint32_t kvStartBatchPerCore[MAX_CORE_NUM] = {0};      // 计算每个core所需搬运的首个kv块对应的batch数量
    uint32_t kvStartOffsetPerCore[MAX_CORE_NUM] = {0};     // 计算每个core所需搬运的首个kv块相较于全量kv初始地址的偏移量
    uint32_t perCoreOutputNumsArr[MAX_CORE_NUM] = {0};     // 记录每个核需要输出的output数量
    uint32_t perCoreStartOutputOffset[MAX_CORE_NUM] = {0}; // 记录每个核输出的首个output偏移量

    NsaCompressTilingData tilingData;
};
// NOTE manually initialize tiling data in hostapi scenario in highest priority template
REGISTER_OPS_TILING_TEMPLATE(NsaCompress, NsaCompressTiling, 0);
} // namespace Nsa
} // namespace optiling
