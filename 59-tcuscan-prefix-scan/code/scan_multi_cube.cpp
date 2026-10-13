/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * @file scan_multi_cube.cpp
 * @brief Kernel implementing a multi-cube inclusive scan.
 */

#include "kernels/constants.h"
#include "kernels/kernel_scan_multi_cube.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_scan_multi_cube.h"

namespace tcuscan {

template <typename InputT>
__aicore__ inline void _run_scan_multi_cube(GM_ADDR input_vec, GM_ADDR lower,
                                            GM_ADDR upper_strict,
                                            GM_ADDR output_vec,
                                            GM_ADDR workspace,
                                            GM_ADDR tilingGm) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  ScanMultiCubeTiling tiling;
  GetTilingData(&tiling, tilingGm);

  const uint32_t vec_len = tiling.vec_len;
  const uint32_t matmul_size = tiling.matmul_size;

  constexpr bool IsInclusive = true;

  const uint64_t available_l2_size = tiling.l2_cache_size;
  const uint32_t fitting_len = scalar::AlignUp(
      available_l2_size / (sizeof(InputT) + sizeof(OutputT)), GM_ALIGNMENT);

  uint32_t remaining_len = vec_len;
  uint32_t offset = 0;
  uint32_t used_size = multi_cube::get_workspace_size<InputT, IsInclusive>(
                           remaining_len, matmul_size) +
                       vec_len * sizeof(InputT) + vec_len * sizeof(OutputT);

  OutputT starting_value = 0;

  while (used_size > available_l2_size && fitting_len < remaining_len) {
    run_scan_multi_cube_kernel<InputT>(
        input_vec + offset * sizeof(InputT), lower, upper_strict,
        output_vec + offset * sizeof(OutputT), workspace, fitting_len,
        matmul_size, starting_value);
    SyncAll<false>();

    remaining_len -= fitting_len;
    used_size = multi_cube::get_workspace_size<InputT, IsInclusive>(
                    remaining_len, matmul_size) +
                vec_len * sizeof(InputT) + vec_len * sizeof(OutputT);
    starting_value = scalar::GetGMValue<OutputT>(
        output_vec + offset * sizeof(OutputT), fitting_len - 1, fitting_len);
    offset += fitting_len;
  }
  if (remaining_len > 0) {
    run_scan_multi_cube_kernel<InputT>(
        input_vec + offset * sizeof(InputT), lower, upper_strict,
        output_vec + offset * sizeof(OutputT), workspace, remaining_len,
        matmul_size, starting_value);
  }
}

}  // namespace tcuscan

/**
 * @brief Run the multi-cube inclusive block scan kernel with dtype fp16
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] lower Pointer to upper-triangular matrix filled with ones.
 * @param [in] upper_strict Pointer to strict upper-triangular matrix filled
 * with ones
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" __global__ __aicore__ void scan_multi_cube_fp16(
    GM_ADDR input_vec, GM_ADDR lower, GM_ADDR upper_strict, GM_ADDR output_vec,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  tcuscan::_run_scan_multi_cube<half>(input_vec, lower, upper_strict,
                                      output_vec, workspace, tiling_gm);
}

/**
 * @brief Call the `scan_multi_cube` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] input_vec Pointer to an input buffer.
 * @param [in] lower Pointer to an input buffer.
 * @param [in] upper_strict Pointer to an input buffer.
 * @param [in] output_vec Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_scan_multi_cube_fp16(uint32_t blockDim, void* stream,
                                            uint8_t* input_vec, uint8_t* lower,
                                            uint8_t* upper_strict,
                                            uint8_t* output_vec,
                                            uint8_t* workspace,
                                            uint8_t* tiling_gm) {
  scan_multi_cube_fp16<<<blockDim, nullptr, stream>>>(
      input_vec, lower, upper_strict, output_vec, workspace, tiling_gm);
}
