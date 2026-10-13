/*
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <fstream>
#include <vector>
#include <gtest/gtest.h>
#include "log/log.h"
#include "platform/platform_infos_def.h"
#include "ut_op_util.h"
#include "kernel_run_context_facker.h"
#include "test_cube_util.h"
#include "exe_graph/runtime/storage_format.h"
#include "exe_graph/runtime/storage_shape.h"

using namespace ut_util;
using namespace std;
using namespace ge;

class AddLayerNormTiling : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "AddLayerNormTiling SetUp" << std::endl;
    }

    static void TearDownTestCase()
    {
        std::cout << "AddLayerNormTiling TearDown" << std::endl;
    }
};

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma_shape = {
        {
            2560,
        },
        {
            2560,
        }};
    gert::StorageShape out_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 0);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_200)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma_shape = {
        {
            2560,
        },
        {
            2560,
        }};
    gert::StorageShape out_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_303)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 11264}, {1, 1024, 11264}};
    gert::StorageShape gamma_shape = {
        {
            11264,
        },
        {
            11264,
        }};
    gert::StorageShape out_shape = {{1, 1024, 11264}, {1, 1024, 11264}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 30);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_302)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 10240}, {1, 1024, 10240}};
    gert::StorageShape gamma_shape = {
        {
            10240,
        },
        {
            10240,
        }};
    gert::StorageShape out_shape = {{1, 1024, 10240}, {1, 1024, 10240}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 20);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_bias_bc)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 10240}, {1, 1024, 10240}};
    gert::StorageShape gamma_shape = {
        {
            10240,
        },
        {
            10240,
        }};
    gert::StorageShape out_shape = {{1, 1024, 10240}, {1, 1024, 10240}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 22);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_bias)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 10240}, {1, 1024, 10240}};
    gert::StorageShape gamma_shape = {
        {
            10240,
        },
        {
            10240,
        }};
    gert::StorageShape out_shape = {{1, 1024, 10240}, {1, 1024, 10240}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &input_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 21);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_slice)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 12288}, {1, 1024, 12288}};
    gert::StorageShape gamma_shape = {
        {
            12288,
        },
        {
            12288,
        }};
    gert::StorageShape out_shape = {{1, 1024, 12288}, {1, 1024, 12288}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &input_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 11);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_bign)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 480}, {1, 1024, 480}};
    gert::StorageShape gamma_shape = {
        {
            480,
        },
        {
            480,
        }};
    gert::StorageShape out_shape = {{1, 1024, 480}, {1, 1024, 480}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &input_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 41);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_sliceext)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 58172}, {1, 1024, 58172}};
    gert::StorageShape gamma_shape = {
        {
            58172,
        },
        {
            58172,
        }};
    gert::StorageShape out_shape = {{1, 1024, 58172}, {1, 1024, 58172}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &input_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 51);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_special_normal)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 40, 5120}, {1, 40, 5120}};
    gert::StorageShape bias_shape = {
        {
            5120,
        },
        {
            5120,
        }};
    gert::StorageShape gamma_shape = {
        {
            5120,
        },
        {
            5120,
        }};
    gert::StorageShape out_shape = {{1, 40, 5120}, {1, 40, 5120}};
    gert::StorageShape rstd_shape = {{1, 40, 1}, {1, 40, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {
        {"SoC_version", "Ascend910B4"}, {"Short_SoC_version", "Ascend910B"}, {"NpuArch", "2201"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &bias_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 62);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_normal310)
{ // network case
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{2, 4096, 320}, {2, 4096, 320}};
    gert::StorageShape gamma_shape = {
        {
            320,
        },
        {
            320,
        }};
    gert::StorageShape out_shape = {{2, 4096, 320}, {2, 4096, 320}};
    gert::StorageShape rstd_shape = {{2, 4096, 1}, {2, 4096, 1}};

    string compile_info_string = R"({
    "hardware_info": {"BT_SIZE": 1024, "load3d_constraints": "0","Intrinsic_data_move_l12ub": true,
                      "Intrinsic_data_move_l0c2ub": true,
                      "UB_SIZE": 262144, "L2_SIZE": 33554432, "L1_SIZE": 1048576,
                      "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 262144, "CORE_NUM": 8}
                      })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    map<string, string> soc_versions = {{"Short_SoC_version", "Ascend310P"}, {"NpuArch", "2202"}};
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", soc_versions);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", soc_versions);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 180);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_mix_input_x1)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 256, 12288}, {1, 256, 12288}};
    gert::StorageShape gamma_shape = {
        {
            12288,
        },
        {
            12288,
        }};
    gert::StorageShape out_shape = {{1, 256, 12288}, {1, 256, 12288}};
    gert::StorageShape rstd_shape = {{1, 256, 1}, {1, 256, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 190);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_mix_input_x2)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 256, 12288}, {1, 256, 12288}};
    gert::StorageShape gamma_shape = {
        {
            12288,
        },
        {
            12288,
        }};
    gert::StorageShape out_shape = {{1, 256, 12288}, {1, 256, 12288}};
    gert::StorageShape rstd_shape = {{1, 256, 1}, {1, 256, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 190);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_mix_input_gamma)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 100, 5120}, {1, 100, 5120}};
    gert::StorageShape gamma_shape = {
        {
            5120,
        },
        {
            5120,
        }};
    gert::StorageShape out_shape = {{1, 100, 5120}, {1, 100, 5120}};
    gert::StorageShape rstd_shape = {{1, 100, 1}, {1, 100, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_BF16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 100);
    //dlog_setlevel(0, 3, 0);
}

// TEST_F(AddLayerNormTiling, add_layer_norm_tiling_single_row_less_tensor_with_bc_bias_fp16_mix_input_gamma) {    //
// tiling key 192
//   //dlog_setlevel(0, 0, 0);
//   gert::StorageShape input_shape = {{1, 100, 12288}, {1, 100, 12288}};
//   gert::StorageShape gamma_shape = {{12288, }, {12288,}};
//   gert::StorageShape out_shape = {{1, 100, 12288}, {1, 100, 12288}};
//   gert::StorageShape rstd_shape = {{1, 100, 1}, {1, 100, 1}};

//   string compile_info_string = R"({
//        "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
//                          "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
//                          "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false, "UB_SIZE":
//                          196608, "L2_SIZE": 33554432, "L1_SIZE": 524288, "L0A_SIZE": 65536, "L0B_SIZE": 65536,
//                          "L0C_SIZE": 131072, "CORE_NUM": 40}
//                          })";
//   map<string, string> soc_infos;
//   map<string, string> aicore_spec;
//   map<string, string> intrinsics;
//   GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

//   // platform info
//   fe::PlatFormInfos platform_info;
//   platform_info.Init();
//   // compile info
//   struct AddLayerNormCompileInfo {};
//   AddLayerNormCompileInfo compile_info;

//   std::string op_type("AddLayerNorm");
//   ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
//   auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
//   auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

//   // tilingParseFunc simulate
//   auto kernel_holder =
//       gert::KernelRunContextFaker()
//           .KernelIONum(2, 1)
//           .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
//           .Outputs({&compile_info})
//           .Build();

//   ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap",
//                                                                                           intrinsics);

//   ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

//   // tilingFunc simulate
//   auto param = gert::TilingData::CreateCap(4096);
//   auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
//   auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
//   ASSERT_NE(param, nullptr);
//   auto holder = gert::TilingContextFaker()
//                     .NodeIoNum(5, 4)
//                     .IrInstanceNum({1, 1, 1, 1, 1})
//                     .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
//                     .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
//                     .CompileInfo(&compile_info)
//                     .PlatformInfo(reinterpret_cast<char*>(&platform_info))
//                     .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
//                     .NodeAttrs({{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.000001)},
//                                 {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
//                     .TilingData(param.get())
//                     .Workspace(ws_size)
//                     .Build();

//   gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
//   ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
//   holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

//   // workspaces nullptr return failed
//   EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
//   // todo check tiling result
//   auto tiling_key = tiling_context->GetTilingKey();
//   ASSERT_EQ(tiling_key, 192);
//   //dlog_setlevel(0, 3, 0);
// }

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_special_reduce_1172)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 640}, {1, 1024, 640}};
    gert::StorageShape gamma_shape = {
        {
            640,
        },
        {
            640,
        }};
    gert::StorageShape out_shape = {{1, 1024, 640}, {1, 1024, 640}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 172);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_special_reduce_82)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 320}, {1, 1024, 320}};
    gert::StorageShape gamma_shape = {
        {
            320,
        },
        {
            320,
        }};
    gert::StorageShape out_shape = {{1, 1024, 320}, {1, 1024, 320}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 82);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_special_reduce_170)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 640}, {1, 1024, 640}};
    gert::StorageShape gamma_shape = {
        {
            640,
        },
        {
            640,
        }};
    gert::StorageShape out_shape = {{1, 1024, 640}, {1, 1024, 640}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(true)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 170);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_special_reduce_80)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 320}, {1, 1024, 320}};
    gert::StorageShape gamma_shape = {
        {
            320,
        },
        {
            320,
        }};
    gert::StorageShape out_shape = {{1, 1024, 320}, {1, 1024, 320}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 80);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_error_shape_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 2560}, {24, 1, 2000}};
    gert::StorageShape gamma_shape = {
        {
            2560,
        },
        {
            2560,
        }};
    gert::StorageShape out_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_error_shape_002)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma_shape = {
        {
            2000,
        },
        {
            2000,
        }};
    gert::StorageShape out_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_error_shape_003)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape gamma_shape = {
        {
            2560,
        },
        {
            2560,
        }};
    gert::StorageShape beta_shape = {
        {
            2000,
        },
        {
            2000,
        }};
    gert::StorageShape out_shape = {{24, 1, 2560}, {24, 1, 2560}};
    gert::StorageShape rstd_shape = {{24, 1, 1}, {24, 1, 1}};

    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 196608, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 40}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &beta_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_fullload_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 512}, {1, 1024, 512}};
    gert::StorageShape gamma_shape = {
        {
            512,
        },
        {
            512,
        }};
    gert::StorageShape out_shape = {{1, 1024, 512}, {1, 1024, 512}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8000);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_fullload_002)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 512}, {1, 1024, 512}};
    gert::StorageShape gamma_shape = {
        {
            512,
        },
        {
            512,
        }};
    gert::StorageShape out_shape = {{1, 1024, 512}, {1, 1024, 512}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &input_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8001);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_fullload_003)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 512}, {1, 1024, 512}};
    gert::StorageShape gamma_shape = {{512}, {512}};
    gert::StorageShape out_shape = {{1, 1024, 512}, {1, 1024, 512}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_infos;
    std::map<std::string, std::string> aicore_spec;
    std::map<std::string, std::string> intrinsics;
    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072, "CORE_NUM": 64
      }
    })";

    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();

    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm")->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm")->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01f)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8002);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_welford_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape gamma_shape = {
        {
            16384,
        },
        {
            16384,
        }};
    gert::StorageShape out_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    string compile_info_string = R"({
       "hardware_info": {"BT_SIZE": 0, "load3d_constraints": "1",
                         "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true, "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
                         "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
                         "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
                         "CORE_NUM": 64}
                         })";
    map<string, string> soc_infos;
    map<string, string> aicore_spec;
    map<string, string> intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    // platform info
    fe::PlatFormInfos platform_info;
    platform_info.Init();
    // compile info
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    std::string op_type("AddLayerNorm");
    ASSERT_NE(gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str()), nullptr);
    auto tiling_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling;
    auto tiling_parse_func = gert::OpImplRegistry::GetInstance().GetOpImpl(op_type.c_str())->tiling_parse;

    // tilingParseFunc simulate
    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();

    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    // tilingFunc simulate
    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holer = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holer.get());
    ASSERT_NE(param, nullptr);
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(4, 4)
                      .IrInstanceNum({1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    holder.GetContext<gert::TilingContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    // workspaces nullptr return failed
    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    // todo check tiling result
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8100);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_welford_002)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape gamma_shape = {{16384}, {16384}};
    gert::StorageShape out_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
      "hardware_info": {
        "BT_SIZE": 0, "load3d_constraints": "1",
        "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
        "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
        "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
        "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
        "CORE_NUM": 64
      }
    })";

    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    auto tiling_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm");
    ASSERT_NE(tiling_impl, nullptr);
    auto tiling_func = tiling_impl->tiling;
    auto tiling_parse_func = tiling_impl->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    // Five inputs, last one has the same shape as the first input
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &input_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8101);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_welford_003)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape gamma_shape = {{16384}, {16384}};
    gert::StorageShape out_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
     "hardware_info": {
     "BT_SIZE": 0, "load3d_constraints": "1",
     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
     "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
     "CORE_NUM": 64
     }
  })";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    auto tiling_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm");
    ASSERT_NE(tiling_impl, nullptr);
    auto tiling_func = tiling_impl->tiling;
    auto tiling_parse_func = tiling_impl->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    // Five inputs, 4th one is gamma_shape, 5th one also gamma_shape
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_SUCCESS);
    auto tiling_key = tiling_context->GetTilingKey();
    ASSERT_EQ(tiling_key, 8102);
    auto block_dim = tiling_context->GetBlockDim();
    ASSERT_EQ(block_dim, 64);
    //dlog_setlevel(0, 3, 0);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_failed_001)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape gamma_shape = {{16384}, {16384}};
    gert::StorageShape out_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
     "hardware_info": {
     "BT_SIZE": 0, "load3d_constraints": "1",
     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
     "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
     "CORE_NUM": 64
     }
  })";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    auto tiling_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm");
    ASSERT_NE(tiling_impl, nullptr);
    auto tiling_func = tiling_impl->tiling;
    auto tiling_parse_func = tiling_impl->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    // Five inputs, 4th one is gamma_shape, 5th one also gamma_shape
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_failed_002)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape gamma_shape = {{16384}, {16384}};
    gert::StorageShape out_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
     "hardware_info": {
     "BT_SIZE": 0, "load3d_constraints": "1",
     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
     "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
     "CORE_NUM": 64
     }
  })";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    auto tiling_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm");
    ASSERT_NE(tiling_impl, nullptr);
    auto tiling_func = tiling_impl->tiling;
    auto tiling_parse_func = tiling_impl->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    // Five inputs, 4th one is gamma_shape, 5th one also gamma_shape
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_failed_003)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape gamma_shape = {{16384}, {16384}};
    gert::StorageShape out_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape rstd_shape = {{1, 1024, 1}, {1, 1024, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    map<string, string> npuarchs = {{"NpuArch", "3510"}};
    std::string compile_info_string = R"({
     "hardware_info": {
     "BT_SIZE": 0, "load3d_constraints": "1",
     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
     "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
     "CORE_NUM": 64
     }
  })";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    auto tiling_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm");
    ASSERT_NE(tiling_impl, nullptr);
    auto tiling_func = tiling_impl->tiling;
    auto tiling_parse_func = tiling_impl->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("version", npuarchs);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    // Five inputs, 4th one is gamma_shape, 5th one also gamma_shape
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT16, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);
    tiling_context->GetPlatformInfo()->SetPlatformRes("version", npuarchs);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}

TEST_F(AddLayerNormTiling, add_layer_norm_tiling_regbase_failed_004)
{
    //dlog_setlevel(0, 0, 0);
    gert::StorageShape input_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape gamma_shape = {{16384}, {16384}};
    gert::StorageShape out_shape = {{1, 1024, 16384}, {1, 1024, 16384}};
    gert::StorageShape rstd_shape = {{1, 512, 1}, {1, 512, 1}};

    std::map<std::string, std::string> soc_version_infos = {{"Short_SoC_version", "Ascend950"}, {"NpuArch", "3510"}};
    std::string compile_info_string = R"({
     "hardware_info": {
     "BT_SIZE": 0, "load3d_constraints": "1",
     "Intrinsic_fix_pipe_l0c2out": false, "Intrinsic_data_move_l12ub": true,
     "Intrinsic_data_move_l0c2ub": true, "Intrinsic_data_move_out2l1_nd2nz": false,
     "UB_SIZE": 245760, "L2_SIZE": 33554432, "L1_SIZE": 524288,
     "L0A_SIZE": 65536, "L0B_SIZE": 65536, "L0C_SIZE": 131072,
     "CORE_NUM": 64
     }
  })";
    std::map<std::string, std::string> soc_infos, aicore_spec, intrinsics;
    GetPlatFormInfos(compile_info_string.c_str(), soc_infos, aicore_spec, intrinsics);

    fe::PlatFormInfos platform_info;
    platform_info.Init();
    struct AddLayerNormCompileInfo {
        uint32_t aivCoreNum_ = 0;
        uint32_t sysWorkspaceSize_ = 0;
        uint64_t ubSize_ = 0;
        uint32_t vecRegSize_ = 0;
        uint32_t blockSize_ = 0;
        bool isAscend950_ = false;
    };
    AddLayerNormCompileInfo compile_info;

    auto tiling_impl = gert::OpImplRegistry::GetInstance().GetOpImpl("AddLayerNorm");
    ASSERT_NE(tiling_impl, nullptr);
    auto tiling_func = tiling_impl->tiling;
    auto tiling_parse_func = tiling_impl->tiling_parse;

    auto kernel_holder =
        gert::KernelRunContextFaker()
            .KernelIONum(2, 1)
            .Inputs({const_cast<char*>(compile_info_string.c_str()), reinterpret_cast<void*>(&platform_info)})
            .Outputs({&compile_info})
            .Build();
    ASSERT_TRUE(kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->Init());
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "AICoreintrinsicDtypeMap", intrinsics);
    kernel_holder.GetContext<gert::TilingParseContext>()->GetPlatformInfo()->SetPlatformRes(
        "version", soc_version_infos);
    ASSERT_EQ(tiling_parse_func(kernel_holder.GetContext<gert::KernelContext>()), ge::GRAPH_SUCCESS);

    auto param = gert::TilingData::CreateCap(4096);
    auto workspace_size_holder = gert::ContinuousVector::Create<size_t>(4096);
    auto ws_size = reinterpret_cast<gert::ContinuousVector*>(workspace_size_holder.get());
    ASSERT_NE(param, nullptr);

    // Five inputs, 4th one is gamma_shape, 5th one also gamma_shape
    auto holder = gert::TilingContextFaker()
                      .NodeIoNum(5, 4)
                      .IrInstanceNum({1, 1, 1, 1, 1})
                      .InputShapes({&input_shape, &input_shape, &gamma_shape, &gamma_shape, &gamma_shape})
                      .OutputShapes({&out_shape, &rstd_shape, &rstd_shape, &out_shape})
                      .CompileInfo(&compile_info)
                      .PlatformInfo(reinterpret_cast<char*>(&platform_info))
                      .NodeInputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeInputTd(4, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(0, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(1, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(2, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeOutputTd(3, ge::DT_FLOAT, ge::FORMAT_ND, ge::FORMAT_ND)
                      .NodeAttrs(
                          {{"epsilon", Ops::NN::AnyValue::CreateFrom<float>(0.01)},
                           {"additional_output", Ops::NN::AnyValue::CreateFrom<bool>(false)}})
                      .TilingData(param.get())
                      .Workspace(ws_size)
                      .Build();

    gert::TilingContext* tiling_context = holder.GetContext<gert::TilingContext>();
    ASSERT_NE(tiling_context->GetPlatformInfo(), nullptr);
    tiling_context->GetPlatformInfo()->SetPlatformRes("SoCInfo", soc_infos);
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreSpec", aicore_spec);
    tiling_context->GetPlatformInfo()->SetCoreNumByCoreType("AICore");
    tiling_context->GetPlatformInfo()->SetPlatformRes("AICoreintrinsicDtypeMap", intrinsics);

    EXPECT_EQ(tiling_func(tiling_context), ge::GRAPH_FAILED);
}
