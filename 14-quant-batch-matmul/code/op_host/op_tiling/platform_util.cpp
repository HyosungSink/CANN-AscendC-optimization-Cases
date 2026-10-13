/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file platform_util.cc
 * \brief function of platform_util
 */

#include <algorithm>
#include <cmath>
#include "platform/platform_info.h"
#include "log/log.h"
#include "platform_util.h"

namespace optiling {
void PlatformUtil::GetLocalMemSize(fe::PlatFormInfos &platform_info, const string &lable, const string &mem_type,
                                      uint64_t &size) {
  std::string temp_str;
  size = 0;
  if (platform_info.GetPlatformRes(lable, mem_type, temp_str)) {
    OP_LOGD("NO_OP_NAME", "PLATFORM INFO %s: %s", mem_type.c_str(), temp_str.c_str());
    try {
      size = atol(temp_str.c_str());
    } catch (const std::exception &e) {
      OP_LOGE_WITHOUT_REPORT("NO_OP_NAME", "illegal PLATFORM INFO %s: %s", mem_type.c_str(), e.what());
    }
  } else {
    OP_LOGW("NO_OP_NAME", "NO PLATFORM INFO for %s", mem_type.c_str());
  }
}

void PlatformUtil::ParseRuntimePlatformInfo(optiling::QuantBatchMatmulV3CompileInfo& compileInfo, const char *op_name, fe::PlatFormInfos &platform_info) {
  (void) op_name;
  platform_info.GetLocalMemSize(fe::LocalMemType::UB, compileInfo.ubSize);
  platform_info.GetLocalMemSize(fe::LocalMemType::L1, compileInfo.l1Size);
  platform_info.GetLocalMemSize(fe::LocalMemType::L2, compileInfo.l2Size);
  platform_info.GetLocalMemSize(fe::LocalMemType::L0_A, compileInfo.l0aSize);
  platform_info.GetLocalMemSize(fe::LocalMemType::L0_B, compileInfo.l0bSize);
  platform_info.GetLocalMemSize(fe::LocalMemType::L0_C, compileInfo.l0cSize);
}
}
