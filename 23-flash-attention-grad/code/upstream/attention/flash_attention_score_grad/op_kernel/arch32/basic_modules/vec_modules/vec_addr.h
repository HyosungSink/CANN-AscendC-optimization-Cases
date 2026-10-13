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
 * \file vec_addr.h
 * \brief
 */
#ifndef _VEC_ADDR_H_
#define _VEC_ADDR_H_
#include "../common_header.h"

namespace VEC_ADDR {
class VecAddr {
public:
    struct VecAddrInfo *globalVecAddr;
    int32_t s1BlockNum;
    int32_t s2BlockNum;
    int32_t s2TailLength;
    int32_t s1TailLength;
    int32_t roundId = 1;
    int32_t g;
    int32_t n2;
    int32_t lastBatchQSum;
    int32_t lastBatchKSum;

    __aicore__ uint64_t getSeqRealLength(int32_t sIdx, int32_t len, int32_t s_block_num, int32_t s_tail)
    {
        if (s_tail > 0 && (sIdx + len == s_block_num)) {
            return (len - 1) * BASE_BLOCK_LENGTH + s_tail;
        } else {
            return len * BASE_BLOCK_LENGTH;
        }
    }

    __aicore__ int64_t getSeqLen(int32_t i, __gm__ uint8_t *seq_Len)
    {
        int64_t actualSeqlen;
        if (i == 0) {
            actualSeqlen = ((__gm__ int64_t *)seq_Len)[0];
        } else {
            actualSeqlen = ((__gm__ int64_t *)seq_Len)[i] - ((__gm__ int64_t *)seq_Len)[i - 1];
        }
        return actualSeqlen;
    }

    __aicore__ int64_t getTotalLen(int32_t i, __gm__ uint8_t *seq_Len)
    {
        int64_t actualTotalSeqlen = ((__gm__ int64_t *)seq_Len)[i];
        return actualTotalSeqlen;
    }

    __aicore__ void blockAdd(int ky)
    {
        int col = (VEC_BLOCK_LIMIT - num) / ky;
        int startS1Id = s1Idx;
        int startS2Id = s2Idx;
        bool ur = true;
        int kx;
        int cnt;

        if (sparseType == 0) {
            if (startS2Id + col < s2BlockNum) {
                s2Idx += col;
            } else {
                col = s2BlockNum - startS2Id;
                s2Idx = 0;
                s1Idx += ky;
            }

            record(startS1Id, startS2Id, col, ky, col * ky, false, lastBatchQSum, lastBatchKSum);
            num += col * ky;
            return;
        }

        if (startS2Id + col == startS1Id + 1) {
            col = (VEC_BLOCK_LIMIT + 1 - num) / ky;
        }

        int reachS2Edge = s2BlockNum < startS1Id + ky;
        int kxMax = reachS2Edge ? s2BlockNum : startS1Id + ky;

        if (col + startS2Id >= kxMax) {
            kx = kxMax - startS2Id;
            s2Idx = 0;
            s1Idx += ky;

            ur = (reachS2Edge || ky == 1);
            cnt = kx * ky - !ur;
        } else {
            kx = col;
            s2Idx += kx;
            cnt = kx * ky;
            if (startS1Id + 1 < startS2Id + kx) {
                ur = false;
            }
        }

        record(startS1Id, startS2Id, kx, ky, cnt, ur, lastBatchQSum, lastBatchKSum);
        num += cnt;
    }

    __aicore__ void getOffset(VecBlockInfo &vecPhyAddr, int32_t s1Id, int32_t s2Id, int32_t blockId, int row, int col)
    {
        vecPhyAddr.batchIdx = batchIdx;
        vecPhyAddr.headNumIdx = headNumIdx;
        vecPhyAddr.S1Idx = s1Id + row;
        vecPhyAddr.S2Idx = s2Id + col;
        vecPhyAddr.n2Idx = headNumIdx / g;
        vecPhyAddr.gIdx = headNumIdx % g;
        vecPhyAddr.offset = blockId * BASE_BLOCK_LENGTH * BASE_BLOCK_LENGTH;
        vecPhyAddr.lengthy = BASE_BLOCK_LENGTH;
        vecPhyAddr.lengthx = BASE_BLOCK_LENGTH;

        if ((row + s1Id == s1BlockNum - 1) && s1TailLength > 0) {
            vecPhyAddr.lengthy = s1TailLength;
        }
        if ((col + s2Id == s2BlockNum - 1) && s2TailLength > 0) {
            vecPhyAddr.lengthx = s2TailLength;
        }
    }

    __aicore__ __inline__ void record(int32_t s1Id, int32_t s2Id, int32_t kx, int32_t ky, int32_t cnt, int32_t ur,
                                      int32_t leftLastBatchSum, int32_t rightLastBatchSum)
    {
        if (coreSegmentBlockNum % coreNum != cubeIdx) {
            return;
        }
        int32_t blockId = num;
        for (int x = 0; x < kx; x++) {
            for (int y = 0; y < ky; y++) {
                if (sparseType == 0 || s1Id + y >= s2Id + x) {
                    getOffset(globalVecAddr->VecBlkInfo[blockId], s1Id, s2Id, blockId, y, x);
                    blockId++;
                }
            }
        }
        globalVecAddr->blockLength += cnt;
    }

    // 分块主逻辑，与cube侧相同
    __aicore__ int32_t addr_mapping(struct VecAddrInfo *vecAddrInfo)
    {
        globalVecAddr = vecAddrInfo;
        globalVecAddr->blockLength = 0;
        int having = 0;

        while (1) {
            if (s1Idx < s1BlockNum && s2Idx < s2BlockNum) {
                for (int lens : lengths) {
                    if ((num + lens) <= VEC_BLOCK_LIMIT && (s1Idx + lens - 1) < s1BlockNum) {
                        if (s2Idx != 0 && lens > s1MaxLength - s1Idx) {
                            continue;
                        }

                        if (s2Idx == 0) {
                            s1MaxLength = s1Idx + lens;
                        }

                        if (lens == ROW_4) {
                            int col = (VEC_BLOCK_LIMIT - num) / lens;
                            int kx;
                            bool ur;

                            int reachS2Edge = s2BlockNum <= s1Idx + 1;
                            int kxMax = (sparseType == 0 || reachS2Edge) ? s2BlockNum : s1Idx + 1;

                            if ((sparseType == 0) || (s2Idx + col < kxMax) || reachS2Edge) {
                                kx = s2Idx + col < kxMax ? col : s2BlockNum - s2Idx;
                                ur = true;
                                record(s1Idx, s2Idx, kx, ROW_2, kx * ROW_2, ur, lastBatchQSum, lastBatchKSum);
                                num += kx * ROW_2;
                                record(s1Idx + ROW_2, s2Idx, kx, ROW_2, kx * ROW_2, ur, lastBatchQSum, lastBatchKSum);
                                num += kx * ROW_2;
                                s2Idx += kx;

                                if (s2Idx >= s2BlockNum) {
                                    s2Idx = 0;
                                    s1Idx += ROW_4;
                                }
                            } else if (s2Idx + col > s1Idx + 1) {
                                kx = s1Idx + ROW_2 - s2Idx;
                                ur = false;
                                record(s1Idx, s2Idx, kx, ROW_2, kx * ROW_2 - 1, ur, lastBatchQSum, lastBatchKSum);
                                num += kx * ROW_2 - 1;
                                s1Idx += ROW_2;

                                blockAdd(ROW_2);
                            } else {
                                kx = (VEC_BLOCK_LIMIT + 1 - num) / lens;
                                int have_ur_block = kx != col;
                                ur = kx == col;
                                record(s1Idx, s2Idx, kx, ROW_2, kx * ROW_2 - have_ur_block, ur, lastBatchQSum, lastBatchKSum);
                                num += kx * ROW_2 - have_ur_block;
                                record(s1Idx + ROW_2, s2Idx, kx, ROW_2, kx * ROW_2, true, lastBatchQSum, lastBatchKSum);
                                num += kx * ROW_2;

                                s2Idx += kx;
                                if (have_ur_block) {
                                    s1Idx += ROW_2;
                                }
                            }
                        } else {
                            blockAdd(lens);
                        }

                        break;
                    }
                }

                if (s1MaxLength - s1Idx + num >= VEC_BLOCK_LIMIT) {
                    num = 0;
                    coreSegmentBlockNum++;
                    if (coreSegmentBlockNum == roundId * coreNum) {
                        break;
                    }
                }

                if (s1Idx >= s1BlockNum && batchIdx >= batchNum - 1 && headNumIdx >= headNum - 1) {
                    return having;
                    break;
                }
            } else {
                s1Idx = 0;
                s2Idx = 0;
                if (headNumIdx == headNum - 1 || s1 == 0 || s2 == 0) {
                    headNumIdx = 0;

                    if (batchIdx < batchNum) {
                        lastBatchQSum = getTotalLen(batchIdx, seqLenQ);
                        lastBatchKSum = getTotalLen(batchIdx, seqLenK);
                        batchIdx += 1;
                        if(batchIdx < batchNum){
                            s1 = getSeqLen(batchIdx, seqLenQ);
                            s2 = getSeqLen(batchIdx, seqLenK);
                        }else{
                            s1 = 0;
                            s2 = 0;
                        }
                        s2TailLength = s2 % BASE_BLOCK_LENGTH;
                        s1TailLength = s1 % BASE_BLOCK_LENGTH;

                        s1BlockNum = (s1 + BASE_BLOCK_LENGTH - 1) / BASE_BLOCK_LENGTH;
                        s2BlockNum = (s2 + BASE_BLOCK_LENGTH - 1) / BASE_BLOCK_LENGTH;
                    } else {
                        break;
                    }
                } else {
                    headNumIdx++;
                }
            }

            if (num > VEC_BLOCK_LIMIT - VEC_COL_LIMIT) {
                num = 0;
                coreSegmentBlockNum++;
                if (coreSegmentBlockNum == roundId * coreNum) {
                    break;
                }
            }
        }
        roundId++;
        return having;
    }

    __aicore__ void init(int32_t batchSize, int32_t heads, int32_t gin, int32_t headDims, int32_t core_index,
                         __gm__ uint8_t *actual_seq_qlen, __gm__ uint8_t *actual_seq_kvlen, int32_t cubeNum,
                         int32_t sparseMode)
    {
        sparseType = sparseMode;
        batchNum = batchSize;
        coreNum = cubeNum;
        headNum = heads;
        headDim = headDims;
        seqLenQ = actual_seq_qlen;
        seqLenK = actual_seq_kvlen;
        g = gin;
        n2 = headNum / g;
        seqLenQ = actual_seq_qlen;
        seqLenK = actual_seq_kvlen;
        cubeIdx = core_index / 2;
        batchIdx = 0;

        while ((s1 == 0 || s2 == 0) && batchIdx < batchNum) {
            s1 = getSeqLen(batchIdx, seqLenQ);
            s2 = getSeqLen(batchIdx, seqLenK);
        }

        s1BlockNum = (s1 + 127) / 128;
        s2BlockNum = (s2 + 127) / 128;
        s2TailLength = s2 % 128;
        s1TailLength = s1 % 128;
        lastBatchQSum = 0;
        lastBatchKSum = 0;
    }

protected:
    __aicore__ void clear_calc_info()
    {
        batchIdx = 0;
        headNumIdx = 0;
        s1Idx = 0;
        s2Idx = 0;
        s1LastMaxLength = 0;
        s1 = 0;
        s2 = 0;
        num = 0;
        lastBlockLength = 0;
    }

private:
    int32_t batchIdx = 0;   // 当前计算到的batch
    int32_t headNumIdx = 0; // 当前计算到的head
    int32_t s1Idx = 0;      // 当前s1方向计算到的位置
    int32_t s2Idx = 0;      // 当前s2方向计算到的位置
    int32_t s1LastMaxLength = 0;
    int32_t s1MaxLength = 0;
    int32_t s1 = 0;              // 当前处理的batch的s1方向sequnce
    int32_t s2 = 0;              // 当前处理的batch的s2方向sequnce
    int32_t num = 0;             // 处理到当前section的累计block数
    int32_t lastBlockLength = 0; // 上个section计算的column数
    int32_t coreSegmentBlockNum = 0;
    int32_t batchNum = 0;
    int32_t coreNum = 0;
    int32_t headNum = 0;
    int32_t headDim = 0;
    int32_t sparseType = 0;

    constexpr static int32_t VEC_BLOCK_LIMIT = 16; // 每个section的处理block上限
    constexpr static int32_t VEC_COL_LIMIT = 4;    // 每个section的处理列上限
    constexpr static int32_t BASE_BLOCK_LENGTH = 128;
    constexpr static int32_t ROW_2 = 2;
    constexpr static int32_t ROW_4 = 4;

    __gm__ uint8_t *seqLenQ;
    __gm__ uint8_t *seqLenK;

    int32_t cubeIdx;
    int32_t lengths[3] = {4, 2, 1};
};
} // namespace VEC_ADDR
#endif