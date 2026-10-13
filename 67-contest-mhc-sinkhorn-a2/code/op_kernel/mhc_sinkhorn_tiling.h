// Tiling结构体定义的头文件
#pragma once

#include <cstdint>

struct MhcSinkhornTilingData {
    uint32_t batch;        // 矩阵总数（T 或 B*S）
    uint32_t n;            // 方阵边长 4/6/8
    uint32_t numIters;     // Sinkhorn 迭代次数
    float    eps;          // 防除零参数
    uint32_t coreNum;      // 启动核数
    uint32_t batchPerCore; // 每核矩阵数（向上取整）
};
