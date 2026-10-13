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

from abc import ABC

import torch

from atk.common.log import Logger
from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.function_api import FunctionApi
from atk.configs.results_config import TaskResult
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
from atk.configs.results_config import OutputData

logging = Logger().get_logger()

try:
    import torch_npu
except Exception as e:
    logging.warning("import torch_npu failed!!!")

@register("aclnn_rms_norm")
class FunctionSegsumApi(FunctionApi):

    def get_format(self, input_data: InputDataset, index=None, name=None):
        return AclFormat.ACL_FORMAT_ND

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        x = input_data.kwargs["input"]
        gamma = input_data.kwargs["gamma"]
        eps = 1e-6
        if self.device in ["cpu", "gpu"]:
            rstd = torch.rsqrt(x.pow(2).mean(tuple(range(len(x.shape) - len(gamma.shape), len(x.shape))), keepdim=True) + 1e-6)
            out = x * rstd
            output = out * gamma
        if self.device == "npu":
            output, rstd = torch_npu.npu_rms_norm(x, gamma, 1e-06)
        return output, rstd.float()
        