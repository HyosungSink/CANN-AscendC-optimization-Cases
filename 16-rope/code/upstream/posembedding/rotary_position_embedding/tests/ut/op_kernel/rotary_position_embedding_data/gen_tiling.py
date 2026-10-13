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
import numpy as np
import sys

# [10, 3, 64] bfloat16_t  interleave
case0_params = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 12, 46, 1, 14, 6, 34, 2, 1, 2, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
# bfloat16_t  rotate
case1_params = [274877906945, 549755814016, 274877907072, 549755814016, 549755813952, 4294967424, 4294967297, 1, 0,
                211106232532992, 32768, 4294967297, 4294967297, 8589934594, 0, 8589934594, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 16, 1, 196608, 1, 24, 128, 64, 24, 64, 128, 128, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0]

case1_64_params = [274877906945, 549755814016, 274877907072, 549755814016, 549755813952, 4294967424, 4294967297, 1, 16,
                211106232532992, 32768, 4294967297, 4294967297, 8589934594, 16, 8589934594, 2, 16, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 16, 1, 196608, 1, 24, 64, 64, 24, 64, 64, 64, 64, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 
                16, 16, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
                0, 0, 0, 0, 0, 0, 0]

case1_32_params = [274877906945, 549755814016, 274877907072, 549755814016, 549755813952, 4294967424, 4294967297, 1, 8,
                211106232532992, 32768, 4294967297, 4294967297, 8589934594, 8, 8589934594, 2, 8, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 16, 1, 102400, 1, 24, 32, 2048, 24, 32, 32, 32, 32, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0]

params_info = {
    "case0": case0_params,
    "case1": case1_params,
    "case1_64": case1_64_params,
    "case1_32": case1_32_params
}

def main():
    params_list = params_info[sys.argv[1]]   # python gen_tiling.py case0  sys.argv[1]="case0"
    base_params = np.array(params_list, dtype=np.int64)
    tiling_file = open("tiling.bin", "wb")
    base_params.tofile(tiling_file)


if __name__ == '__main__':
    main()
