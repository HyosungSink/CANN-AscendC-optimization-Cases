// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

// mHC-expand 算子 Tiling 数据（DESIGN.md §2）
// 布局注意：全部 uint32_t 集中在前，避免与 int64_t 混排产生隐式 padding；
// 整体按 8 字节对齐读取，与 GET_TILING_DATA_WITH_STRUCT 的读取假设一致。
struct MhcExpandTilingData {
    uint32_t S;            // token 数（前向输入行数 / 反向输出行数）
    uint32_t D;            // 隐藏维
    uint32_t m;            // mhc_mult 扩展倍数
    uint32_t backward;     // 0 = 前向(复制广播)，1 = 反向(求和归约)

    uint32_t blockDim;     // 实际启动核数
    uint32_t rowsPerCore;  // 每核负责的任务数（任务含义随 splitMode 变化）
    uint32_t tailRows;     // 最后一个核的任务数（信息字段，kernel 用区间计算更稳）

    uint32_t dTileLen;     // D 方向单 tile 元素数（32B 对齐，典型 512~2048；整行模式 = D）
    uint32_t dTileNum;     // D 方向 tile 数 = ceildiv(D, dTileLen)
    uint32_t dTailLen;     // 最后一个 D tile 的有效元素数

    uint32_t splitMode;    // 0 = ROW_SPLIT, 1 = ROW_STREAM_SPLIT(前向), 2 = ELEMENT_SPLIT
};
