#!/usr/bin/env python3
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

import numpy as np
import torch
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi

try:
    import torch_npu
except Exception as e:
    logging.warning("import torch_npu failed!!!")


def compute_golden(input_x1, input_x2, input_beta, input_gamma, biasOptional,input_dtype,
                   additional=False, epsilon=1e-05, reduce_axis=-1):
    if "fp16" in str(input_dtype) or "bf16" in str(input_dtype):
        np_data_type = np.float32
        x1_hp = input_x1.astype(np_data_type)
        x2_hp = input_x2.astype(np_data_type)
        beta_hp = input_beta.astype(np_data_type)
        gamma_hp = input_gamma.astype(np_data_type)
        biasOptional_hp = biasOptional.astype(np_data_type)
       
    else:
        x1_hp = input_x1
        x2_hp = input_x2
        beta_hp = input_beta
        gamma_hp = input_gamma 
        biasOptional_hp = biasOptional       

    input_x = np.add(x1_hp, x2_hp)   

    input_x = np.add(input_x, biasOptional_hp) 
    input_mean = np.mean(input_x, reduce_axis, keepdims=True)
    input_var = np.var(input_x, reduce_axis, keepdims=True)
    input_rstd = 1.0 / np.sqrt(input_var + epsilon)

    input_y = np.subtract(input_x, input_mean) * input_rstd
    input_y = input_y * gamma_hp + beta_hp

    if "bf16" in str(input_dtype):
        tensor_y = torch.from_numpy(input_y.astype(np.float32)).to(torch.bfloat16)
        tensor_x = torch.from_numpy(input_x.astype(np.float32)).to(torch.bfloat16)
    elif "float16" in str(input_dtype):
        tensor_y = torch.from_numpy(input_y.astype(np.float32)).to(torch.float16)
        tensor_x = torch.from_numpy(input_x.astype(np.float32)).to(torch.float16)
    else:
        tensor_y = torch.from_numpy(input_y.astype(np.float32))
        tensor_x = torch.from_numpy(input_x.astype(np.float32))
    
    tensor_mean = torch.from_numpy(input_mean.astype(np.float32))
    tensor_rstd = torch.from_numpy(input_rstd.astype(np.float32))
    if additional:
        return tensor_y, tensor_mean, tensor_rstd, tensor_x
    else:
        return tensor_y, tensor_mean, tensor_rstd


@register("aclnn_add_layer_norm")
# 没有pro的判断 也没有with_output
class FunctionApi(BaseApi):    

    def __call__(self, input_data: InputDataset, with_output: bool = False):        
        if self.device == "npu":
           import torch_npu
           torch_npu.npu.set_compile_mode(jit_compile=False)

        input_x1 = input_data.kwargs["x1"]
        input_x2 = input_data.kwargs["x2"]
        gamma = input_data.kwargs["gamma"]
        beta = input_data.kwargs["beta"]
        biasOptional = input_data.kwargs["biasOptional"]
        additional_output = input_data.kwargs["additional_output"]
        dtype = input_x1.dtype    

        eps = 1e-6

        if self.device == "cpu" or self.device == "gpu":
            return compute_golden(input_x1.detach().clone().cpu().numpy(),
                       input_x2.detach().clone().cpu().numpy(),
                       beta.detach().clone().cpu().numpy(),
                       gamma.detach().clone().cpu().numpy(),biasOptional.detach().clone().cpu().numpy(),
                       dtype, additional_output, eps)            
        else:
            import torch_npu
            y, mean, rstd, x = torch_npu.npu_add_layer_norm(
                x1=input_x1, x2=input_x2, gamma=gamma, beta=beta, epsilon=eps, additional_output=additional_output)

            if additional_output:
                return y, mean, rstd, x
            else:
                return y, mean, rstd

    def init_by_input_data(self, input_data: InputDataset):
        if self.device == "aclnn":
            input_data.kwargs['eps'] = 1e-6
