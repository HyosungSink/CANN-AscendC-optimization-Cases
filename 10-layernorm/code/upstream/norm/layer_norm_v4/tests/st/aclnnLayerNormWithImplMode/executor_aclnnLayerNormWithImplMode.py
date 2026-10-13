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

import torch
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensorStruct, AclFormat

@register("aclnn_layer_norm_with_impl_mode")
class AclnnLayerNormWithImplMode(BaseApi):
    def __call__(self, input_data: InputDataset, with_output: bool = False):

        origin_dtype = input_data.kwargs["input"].dtype
        input = input_data.kwargs["input"]
        normalized_shape = input_data.kwargs["normalizedShape"]
        weight = input_data.kwargs["weightOptional"]
        bias = input_data.kwargs["biasOptional"]
        eps = input_data.kwargs["eps"]
        if self.device == 'cpu' and (input.dtype == torch.float16 or input.dtype == torch.bfloat16):
            res = torch.native_layer_norm(input.to(torch.float32), normalized_shape,
                        weight.to(torch.float32), bias.to(torch.float32), eps)
            output = (res[0].to(origin_dtype),res[1].to(origin_dtype),res[2].to(origin_dtype))
            
        else:
            output = torch.native_layer_norm(input, normalized_shape, weight, bias, eps)
        
        return output

@register("aclnn_layer_norm_with_impl_mode_aclnn")
class AclnnLayerNormWithImplMode(AclnnBaseApi):

    def init_by_input_data(self, input_data):
        input_args, output_packages = super().init_by_input_data(input_data)
        input_args.append(input_args.pop(5))
        self.input_args=input_args
        return input_args, output_packages

    def after_call(self, output_packages):
        output = []
        self.input_args.pop()
        output_packages = self.input_args[-len(output_packages):]
        for output_pack in output_packages:
            output.append(self.acl_tensor_to_torch(output_pack))
        return output