/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file scan_multi_core.cpp
 * @brief Kernel implementing a multi-core inclusive scan using cube-vector
 * approach.
 */

#include "kernels/constants.h"
#include "kernels/kernel_scan_multi_core.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_scan_multi_core.h"

namespace tcuscan {

template <typename InputT>
__aicore__ inline void _run_scan_multi_core_no_l2_split(GM_ADDR input_vec,
                                                        GM_ADDR output_vec,
                                                        GM_ADDR workspace,
                                                        GM_ADDR tilingGm) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  MultiCoreScanTiling tiling;
  GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t matmul_size = tiling.matmul_size;
  constexpr bool IsInclusive = true;

  GM_ADDR const lower = load_tril_matrix<InputT>(matmul_size);

  constexpr OutputT starting_value = 0;
  run_scan_multi_core_kernel<InputT, IsInclusive>(input_vec, lower, output_vec,
                                                  workspace, vec_len,
                                                  matmul_size, starting_value);
}

template <typename InputT>
__aicore__ inline void _run_scan_multi_core(GM_ADDR input_vec,
                                            GM_ADDR output_vec,
                                            GM_ADDR workspace,
                                            GM_ADDR tilingGm) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  MultiCoreScanTiling tiling;
  GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t matmul_size = tiling.matmul_size;
  constexpr bool IsInclusive = true;

  GM_ADDR const lower = load_tril_matrix<InputT>(matmul_size);

  const uint64_t available_l2_size = tiling.l2_cache_size;
  const uint32_t fitting_len = scalar::AlignUp(
      available_l2_size / (sizeof(InputT) + sizeof(OutputT)), GM_ALIGNMENT);

  uint32_t remaining_len = vec_len;
  uint32_t offset = 0;
  uint32_t used_size =
      mc_scan::get_workspace_size<InputT, OutputT, IsInclusive>(remaining_len,
                                                                matmul_size) +
      vec_len * sizeof(InputT) + vec_len * sizeof(OutputT);

  OutputT starting_value = 0;

  while (used_size > available_l2_size && fitting_len < remaining_len) {
    run_scan_multi_core_kernel<InputT, IsInclusive>(
        input_vec + offset * sizeof(InputT), lower,
        output_vec + offset * sizeof(OutputT), workspace, fitting_len,
        matmul_size, starting_value);
    SyncAll<false>();

    remaining_len -= fitting_len;
    used_size = mc_scan::get_workspace_size<InputT, OutputT, IsInclusive>(
                    remaining_len, matmul_size) +
                vec_len * sizeof(InputT) + vec_len * sizeof(OutputT);
    starting_value = scalar::GetGMValue<OutputT>(
        output_vec + offset * sizeof(OutputT), fitting_len - 1, fitting_len);
    offset += fitting_len;
  }
  if (remaining_len) {
    run_scan_multi_core_kernel<InputT, IsInclusive>(
        input_vec + offset * sizeof(InputT), lower,
        output_vec + offset * sizeof(OutputT), workspace, remaining_len,
        matmul_size, starting_value);
  }
}

}  // namespace tcuscan

/**
 * @brief Run the multi core inclusive scan kernel with input dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core_fp16(GM_ADDR input_vec,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::_run_scan_multi_core<half>(input_vec, output_vec, workspace,
                                      tilingGm);
}

/**
 * @brief Run the multi core inclusive scan kernel with input dtype int8
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core_int8(GM_ADDR input_vec,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::_run_scan_multi_core<int8_t>(input_vec, output_vec, workspace,
                                        tilingGm);
}

/**
 * @brief Run the multi core inclusive scan kernel with input dtype fp32
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core_fp32(GM_ADDR input_vec,
                                                           GM_ADDR output_vec,
                                                           GM_ADDR workspace,
                                                           GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::_run_scan_multi_core<float>(input_vec, output_vec, workspace,
                                       tilingGm);
}

/**
 * @brief Run the multi core inclusive scan kernel with input dtype fp16 without
 * L2 splitting optimization.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core_fp16_no_l2(
    GM_ADDR input_vec, GM_ADDR output_vec, GM_ADDR workspace,
    GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::_run_scan_multi_core_no_l2_split<half>(input_vec, output_vec,
                                                  workspace, tilingGm);
}

/**
 * @brief Run the multi core inclusive scan kernel with input dtype int8 without
 * L2 splitting optimization.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core_int8_no_l2(
    GM_ADDR input_vec, GM_ADDR output_vec, GM_ADDR workspace,
    GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::_run_scan_multi_core_no_l2_split<int8_t>(input_vec, output_vec,
                                                    workspace, tilingGm);
}

/**
 * @brief Run the multi core inclusive scan kernel with input dtype fp32 without
 * L2 splitting optimization.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_core_fp32_no_l2(
    GM_ADDR input_vec, GM_ADDR output_vec, GM_ADDR workspace,
    GM_ADDR tilingGm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::_run_scan_multi_core_no_l2_split<float>(input_vec, output_vec,
                                                   workspace, tilingGm);
}

/**
 * @brief Call the `scan_multi_core` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_multi_core_fp16(uint32_t blockDim, void* stream,
                                            uint8_t* input_vec,
                                            uint8_t* output_vec,
                                            uint8_t* workspace,
                                            uint8_t* tilingGm) {
  scan_multi_core_fp16<<<blockDim, nullptr, stream>>>(input_vec, output_vec,
                                                      workspace, tilingGm);
}

/**
 * @brief Call the `scan_multi_core` kernel for INT8 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_multi_core_int8(uint32_t blockDim, void* stream,
                                            uint8_t* input_vec,
                                            uint8_t* output_vec,
                                            uint8_t* workspace,
                                            uint8_t* tilingGm) {
  scan_multi_core_int8<<<blockDim, nullptr, stream>>>(input_vec, output_vec,
                                                      workspace, tilingGm);
}

/**
 * @brief Call the `scan_multi_core` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_multi_core_fp32(uint32_t blockDim, void* stream,
                                            uint8_t* input_vec,
                                            uint8_t* output_vec,
                                            uint8_t* workspace,
                                            uint8_t* tilingGm) {
  scan_multi_core_fp32<<<blockDim, nullptr, stream>>>(input_vec, output_vec,
                                                      workspace, tilingGm);
}

/**
 * @brief Launch the `scan_multi_core_fp16_no_l2` kernel.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_multi_core_fp16_no_l2(
    uint32_t blockDim, void* stream, uint8_t* input_vec, uint8_t* output_vec,
    uint8_t* workspace, uint8_t* tilingGm) {
  scan_multi_core_fp16_no_l2<<<blockDim, nullptr, stream>>>(
      input_vec, output_vec, workspace, tilingGm);
}

/**
 * @brief Launch the `scan_multi_core_int8_no_l2` kernel.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_multi_core_int8_no_l2(
    uint32_t blockDim, void* stream, uint8_t* input_vec, uint8_t* output_vec,
    uint8_t* workspace, uint8_t* tilingGm) {
  scan_multi_core_int8_no_l2<<<blockDim, nullptr, stream>>>(
      input_vec, output_vec, workspace, tilingGm);
}

/**
 * @brief Launch the `scan_multi_core_fp32_no_l2` kernel.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tilingGm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_multi_core_fp32_no_l2(
    uint32_t blockDim, void* stream, uint8_t* input_vec, uint8_t* output_vec,
    uint8_t* workspace, uint8_t* tilingGm) {
  scan_multi_core_fp32_no_l2<<<blockDim, nullptr, stream>>>(
      input_vec, output_vec, workspace, tilingGm);
}
