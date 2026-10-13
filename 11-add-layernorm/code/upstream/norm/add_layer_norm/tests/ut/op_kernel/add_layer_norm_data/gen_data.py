#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
import sys
import os
import numpy as np
import re
import torch
import tensorflow as tf


def parse_str_to_shape_list(shape_str):
    shape_str = shape_str.strip('(').strip(')')
    shape_list = [int(x) for x in shape_str.split(",")]
    return np.array(shape_list), shape_list

def add_layer_norm(input_x1, input_x2, input_gamma, input_beta, input_bias, epsilon = 1e-05, output_x=True):
    x1 = input_x1.astype(np.float32)
    x2 = input_x2.astype(np.float32)
    gamma = input_gamma.astype(np.float32)
    beta = input_beta.astype(np.float32)
    if input_bias is None:         
        x = x1 + x2     
    else:         
        bias = input_bias.astype(np.float32)         
        x = x1 + x2 + bias      
        reduce_axis = -1     
        input_mean = np.mean(x, reduce_axis, keepdims=True)     
        input_var = np.mean(np.power((x - input_mean), 2), reduce_axis, keepdims=True)     
        input_rstd = 1.0 / np.sqrt(input_var + epsilon)      
        y = np.subtract(x, input_mean) * input_rstd     
        y = y * gamma + beta      
        if input_x1.dtype != input_x2.dtype:         
            out_dtype = np.float32     
        else:         
            out_dtype = input_x1.dtype      
        if output_x:         
            return y.astype(out_dtype), input_mean, input_rstd, x.astype(out_dtype)     
        else:         
            return y.astype(out_dtype), input_mean, input_rstd
        
def gen_data_and_golden(input_x_shape_str, input_gamma_shape_str, d_type="float32"):
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16_t": tf.bfloat16.as_numpy_dtype
    }
    np_type = d_type_dict[d_type]
    input_x_shape, _ = parse_str_to_shape_list(input_x_shape_str)
    input_gamma_shape, _ = parse_str_to_shape_list(input_gamma_shape_str)

    input_x_size = np.prod(input_x_shape)
    input_gamma_size = np.prod(input_gamma_shape)
    tmp_input_x1 = np.random.random(input_x_size).reshape(input_x_shape).astype(np_type)
    tmp_input_x2 = np.random.random(input_x_size).reshape(input_x_shape).astype(np_type)
    tmp_input_bias = np.random.random(input_x_size).reshape(input_x_shape).astype(np_type)

    tmp_input_gamma = np.random.random(input_gamma_size).reshape(input_gamma_shape).astype(np_type)
    tmp_input_beta = np.random.random(input_gamma_size).reshape(input_gamma_shape).astype(np_type)
    
    y_golden, mean_golden, rstd_golden, x_golden= add_layer_norm(tmp_input_x1, tmp_input_x2, tmp_input_gamma, tmp_input_beta, tmp_input_bias)
    tmp_y_golden = np.array(y_golden).astype(np_type)
    tmp_mean_golden = np.array(mean_golden).astype(np.float32)
    tmp_rstd_golden = np.array(rstd_golden).astype(np.float32)
    tmp_x_golden = np.array(x_golden).astype(np_type)
    
    # input
    tmp_input_x1.astype(np_type).tofile(f"{d_type}_input_x1.bin")
    tmp_input_x2.astype(np_type).tofile(f"{d_type}_input_x2.bin")
    tmp_input_bias.astype(np_type).tofile(f"{d_type}_input_bias.bin")
    tmp_input_gamma.astype(np_type).tofile(f"{d_type}_input_gamma.bin")
    tmp_input_beta.astype(np_type).tofile(f"{d_type}_input_beta.bin")
    # golden out
    tmp_y_golden.astype(np_type).tofile(f"{d_type}_golden_y.bin")
    tmp_mean_golden.astype(np.float32).tofile(f"{d_type}_golden_mean.bin")
    tmp_rstd_golden.astype(np.float32).tofile(f"{d_type}_golden_rstd.bin")
    tmp_x_golden.astype(np_type).tofile(f"{d_type}_golden_x.bin")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Param num must be 4, actually is ", len(sys.argv))
        exit(1)
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(sys.argv[1], sys.argv[2], sys.argv[3])