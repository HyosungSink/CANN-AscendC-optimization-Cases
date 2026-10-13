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
#

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclFormat
import numpy as np



@register("aclnn_add_rms_norm_quant")
class FunctionRmsNormGradApi(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["epsilon"] = 0.000001
        input_data.kwargs["scale2"] = None
        input_data.kwargs["zeropoint2"] = None
        input_data.kwargs["divMode"] = True

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        
        input_x1 = input_data.kwargs["x1"]
        input_x2 = input_data.kwargs["x2"]
        input_gamma = input_data.kwargs["gamma"]
        input_scales1 = input_data.kwargs["scale1"]
        input_scales2 = input_data.kwargs["scale2"]
        input_zero_points1 = input_data.kwargs["zeropoint1"]
        input_zero_points2 = input_data.kwargs["zeropoint2"]
        epsilon = input_data.kwargs["epsilon"]
        axis = input_data.kwargs["axis"]
        divMode = input_data.kwargs["divMode"]
        kernelType = input_data.kwargs["kernelType"]
        if self.device == "cpu":
            output1, output2, output3 = self.npu_add_rms_norm_quant_golden(input_x1, input_x2, input_gamma, input_scales1,
                                      input_zero_points1, None, None, None, axis, epsilon, True, kernelType)
        output2.zero_()
        return output1, output2, output3
    
    def npu_add_rms_norm_quant_golden(self, input_x1, input_x2, input_gamma, input_scales1,
                                      input_zero_points1, input_scales2=None, input_zero_points2=None, input_beta=None, axis=None ,epsilon=0.000001, divMode=True, kernelType = 1):
        torchType32 = torch.float32
        len_shape_x = len(input_x1.shape)
        x_shape_ori = input_x1.shape
        len_shape_gamma = len(input_gamma.shape)

        n = len(input_x1.shape) - len(input_gamma.shape)
        input_gamma = input_gamma.reshape(np.multiply.reduce(np.array(input_gamma.shape)))

        input_scales1 = input_scales1.reshape(np.multiply.reduce(np.array(input_scales1.shape)))
        if input_scales2 is not None:
            input_scales2 = input_scales2.reshape(np.multiply.reduce(np.array(input_scales2.shape)))
        if input_zero_points1 is not None:
            input_zero_points1 = input_zero_points1.reshape(np.multiply.reduce(np.array(input_zero_points1.shape)))
        if input_zero_points2 is not None:
            input_zero_points2 = input_zero_points2.reshape(np.multiply.reduce(np.array(input_zero_points2.shape)))
        if input_beta is not None:
            input_beta = input_beta.reshape(np.multiply.reduce(np.array(input_beta.shape)))

        x1_shape = input_x1.shape[0:n] + input_gamma.shape
        input_x1 = input_x1.reshape(x1_shape)
        input_x2 = input_x2.reshape(x1_shape)

        len_shape_x = len(input_x1.shape)
        len_shape_gamma = len(input_gamma.shape)
        axis = len_shape_x - len_shape_gamma

        if kernelType == 1:
          input_x_dtype = torch.float16
          input_scales_dtype = torch.float32
        else:
          input_x_dtype = torch.bfloat16
          input_scales_dtype = torch.bfloat16
        
        input_x1 = input_x1.to(input_x_dtype)
        input_x2 = input_x2.to(input_x_dtype)
        input_gamma = input_gamma.to(input_x_dtype)
        
        if input_beta is not None:
          input_beta = input_beta.to(input_x_dtype)
        input_scales1 = input_scales1.type(torchType32)
        if input_scales2 is not None:
            input_scales2 = input_scales2.type(torchType32)
        if input_zero_points1 is not None:
            input_zero_points1 = input_zero_points1.type(torchType32) 
        if input_zero_points2 is not None:
            input_zero_points2 = input_zero_points2.type(torchType32)
        
        # add
        if (input_x_dtype == torch.float16):
            add_x = (input_x1 + input_x2)
        else:
            add_x = (input_x1.type(torchType32) + input_x2.type(torchType32))
            if input_beta is not None:
              add_x = add_x.to(torch.bfloat16)
        # rmsNorm
        x_fp32 = add_x.type(torchType32)
        variance = torch.mean(torch.pow(x_fp32, 2), axis=axis, keepdims=True)
        std = torch.sqrt(variance + epsilon)
        rstd = 1 / std
        result_mid = x_fp32 * rstd

        if input_x_dtype == torch.bfloat16:
            input_gamma_fp32 = input_gamma.type(torchType32)
            y_array = result_mid * input_gamma_fp32
        elif input_x_dtype == torch.float16:
            result_mid_fp16 = result_mid.type(torch.float16)
            y_array = result_mid_fp16 * input_gamma
        
        if input_beta is not None:
          y_array = y_array.type(torchType32) + input_beta.type(torchType32)
        
        # quant
        y = y_array.type(torch.float32)
        tensor_scales1 = input_scales1.to(torch.float32)
        
        min_val = -2**7
        max_val = 2**7 - 1

        # do quant1
        if divMode:
            do_scales1_zeoPoints1 = torch.div(y, tensor_scales1)
        else:
            do_scales1_zeoPoints1 = torch.mul(y, tensor_scales1)
        if input_zero_points1 is not None:
            do_scales1_zeoPoints1 = torch.add(do_scales1_zeoPoints1, input_zero_points1)
        y1_round = torch.round(do_scales1_zeoPoints1)
        y1_to_fp32 = y1_round.to(torch.int32)
        y1_to_fp16 = y1_to_fp32.to(torch.float16)
        y1_trunc = torch.trunc(y1_to_fp16)
        y1_clamp = torch.clamp(y1_trunc, min_val, max_val)

        # do quant2
        if input_scales2 is not None:
            if divMode:
                do_scales2_zero_points2 = torch.div(y, input_scales2)
            else:
                do_scales2_zero_points2 = torch.mul(y, input_scales2)
            if input_zero_points2 is not None:
                do_scales2_zero_points2 = torch.add(do_scales2_zero_points2, input_zero_points2)
            
            y2_round = torch.round(do_scales2_zero_points2)
            y2_to_fp32 = y2_round.to(torch.int32)
            y2_to_fp16 = y2_to_fp32.to(torch.float16)
            y2_trunc = torch.trunc(y2_to_fp16)
            y2_clamp = torch.clamp(y2_trunc, min_val, max_val)
        else:
            y2_clamp = torch.zeros(x_shape_ori)

        return y1_clamp.type(torch.int8).reshape(x_shape_ori), y2_clamp.type(torch.int8).reshape(x_shape_ori), add_x.type(input_x_dtype).reshape(x_shape_ori)
        

@register("sample_aclnn_api")
class SampleAclnnApi(AclnnBaseApi):  # SampleApi类型仅需设置唯一即可。
    def init_by_input_data(self, input_data: InputDataset):
        del input_data.kwargs["kernelType"]
        input_args, output_packages = super().init_by_input_data(input_data)
        return input_args, output_packages
    # pyaclnn的自定义api
    def after_call(self, output_packages):
        output1, output2, output3 = super().after_call(output_packages)
        output2.zero_()
        return output1, output2, output3

