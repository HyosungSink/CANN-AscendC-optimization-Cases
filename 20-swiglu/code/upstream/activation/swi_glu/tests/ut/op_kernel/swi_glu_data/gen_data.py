#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import numpy as np
import torch


def do_swiglu(a,b):

    y = torch.sigmoid(a) * a * b

    return y

def gen_golden_data_simple(n, l, dtype):
    # np.random.seed(1)
    data_x = np.random.uniform(-2, 2, [int(n), int(l)]).astype(dtype)
    tensor_x = torch.from_numpy(data_a)

    a, b = tensor_x.chunk(2, dim=-1)

    a = a.to(torch.float32)
    b = b.to(torch.float32)
    y = do_gelu(a,b)
    y = y.to(torch.float16)

    res = y.numpy()

    data_x.tofile("./input_x.bin")
    res.tofile("./output_y_golden.bin")


if __name__ == "__main__":
    gen_golden_data_simple(sys.argv[1], sys.argv[2], sys.argv[3])