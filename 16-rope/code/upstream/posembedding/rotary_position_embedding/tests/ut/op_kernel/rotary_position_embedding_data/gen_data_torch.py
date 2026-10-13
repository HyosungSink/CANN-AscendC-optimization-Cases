#!/usr/bin/python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys
import os
import numpy as np
import torch

dtype_map = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32
}


def get_interleave_matrix(n, dtype_q):
    matrix = torch.zeros(n, n, dtype=dtype_q)
    for i in range(0, n, 2):
        matrix[i + 0, i + 1] = 1
        matrix[i + 1, i + 0] = -1
    return matrix


def get_half_matrix(n, dtype_q):
    matrix = torch.zeros(n, n, dtype=dtype_q)
    half = n // 2
    matrix[:half, half:] = torch.eye(half)
    matrix[half:, :half] = -torch.eye(half)
    return matrix


def gen_data_and_golden(a, b, c, d, dtype_q):
    data_x = torch.FloatTensor(int(a), int(b), int(c), int(d)).uniform_(-2, 2).to(dtype_q)
    data_cos = torch.FloatTensor(1, 1, int(c), int(d)).uniform_(-3, 4).to(dtype_q)
    data_sin = torch.FloatTensor(1, 1, int(c), int(d)).uniform_(-4, 5).to(dtype_q)
    data_rotate_half = get_half_matrix(int(d), dtype_q)
    data_rotate_inter = get_interleave_matrix(int(d), dtype_q)

    torch.save(data_x, "./x.pt")
    torch.save(data_cos, "./cos.pt")
    torch.save(data_sin, "./sin.pt")
    torch.save(data_sin, "./rotate_half.pt")
    torch.save(data_sin, "./rotate_inter.pt")

if __name__ == "__main__":
    os.system("rm -rf *.bin")
    if sys.argv[5] not in dtype_map:
        raise ValueError(f"不支持的 dtype：{sys.argv[5]}，支持的类型：{list(dtype_map.keys())}")
    dtype_q = dtype_map[sys.argv[5]]
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], dtype_q)
    exit(0)