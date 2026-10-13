/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_multi_core.h
 * @brief Kernel implementing different variants of a scan operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "kernel_complete_rows.h"
#include "kernel_pad.h"
#include "kernel_reduce_tiles.h"
#include "kernel_row_scan.h"
#include "tcuscan_utils.h"

using namespace AscendC;
using namespace tcuscan;

namespace mc_scan {

/**
 * @brief Calculate the workspace size for multi core scan.
 *
 * @tparam InputT Input data type.
 * @tparam OutputT Output data type.
 *
 * @param [in] input_elems Number of elements in the input vector.
 * @param [in] matmul_size Size of the matmul used in scan.
 * @return Size of the workspace in bytes.
 */
template <typename InputT, typename OutputT, bool IsInclusive = true>
__aicore__ inline uint32_t get_workspace_size(uint32_t input_elems,
                                              uint32_t matmul_size) {
  const uint32_t align_size = matmul_size * matmul_size;
  const uint32_t padded_input_len = scalar::AlignUp(input_elems, align_size);

  const uint32_t padded_input_size = padded_input_len * sizeof(InputT);
  const uint32_t padded_rowwise_size = padded_input_len * sizeof(OutputT);
  const uint32_t sums_len = GetBlockNum() * GetTaskRation();
  const uint32_t sums_size =
      scalar::AlignUp(sums_len * sizeof(OutputT), GM_ALIGNMENT);

  if (padded_input_len == input_elems) {
    if constexpr (IsInclusive)
      return sums_size;
    else
      return padded_rowwise_size + sums_size;
  } else
    return padded_input_size + padded_rowwise_size + sums_size;
}

}  // namespace mc_scan

/**
 * @brief Run the multi core scan kernel.
 *
 * @tparam IsInclusive Indicates whether the scan is inclusive or exclusive.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] upper_triangular Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] vec_len Number of elements to process.
 * @param [in] matmul_size Size of the matmul tile used in the kernel.
 * @param [in] starting_value Starting value added to all entries of the
 * output vector.
 */

template <typename InputT, bool IsInclusive = true>
__aicore__ inline void run_scan_multi_core_kernel(
    GM_ADDR input_vec, GM_ADDR upper_triangular, GM_ADDR output_vec,
    GM_ADDR workspace, uint32_t vec_len, uint32_t matmul_size,
    cube_unit::CubeOutType_t<InputT> starting_value = 0) {
  using OutputT = cube_unit::CubeOutType_t<InputT>;

  const uint32_t align_size = matmul_size * matmul_size;
  const uint32_t padded_vec_len_c = scalar::AlignUp(vec_len, align_size);
  const uint32_t padded_vec_len_v = scalar::AlignUp(vec_len, align_size / 2);

  const uint32_t padded_input_size = padded_vec_len_c * sizeof(InputT);
  const uint32_t padded_rowwise_size = padded_vec_len_c * sizeof(OutputT);

  if (vec_len % align_size || vec_len % UB_ALIGNMENT) {
    GM_ADDR const padded_input = workspace;
    GM_ADDR const padded_rowwise_scan = padded_input + padded_input_size;
    GM_ADDR const sums = padded_rowwise_scan + padded_rowwise_size;

    tcuscan::run_pad_kernel<InputT, false>(input_vec, padded_input, vec_len,
                                           align_size);

    AscendC::SyncAll<false /*isAIVOnly*/>();
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    PipeBarrier<PIPE_ALL>();

    if ASCEND_IS_AIC {
      tcuscan::KernelRowScan<InputT> op_cube(matmul_size, matmul_size,
                                             padded_vec_len_c);
      op_cube.Init(padded_input, upper_triangular, padded_rowwise_scan);
      op_cube.Process();
      PipeBarrier<PIPE_ALL>();
    }

    if ASCEND_IS_AIV {
      tcuscan::KernelReduceTiles<InputT> op_reduce(
          matmul_size * matmul_size / 2, padded_vec_len_v);
      op_reduce.Init(padded_input, sums);
      op_reduce.Process();
    }

    SyncAll<false /*isAIVOnly*/>();

    if ASCEND_IS_AIV {
      tcuscan::KernelCompleteRows<OutputT, IsInclusive> op(
          matmul_size, matmul_size / 2, vec_len);
      op.Init(padded_rowwise_scan, sums, output_vec);
      op.Process(starting_value);
    }
  } else {
    // If possible, we try to execute phase 2 in place, as it might
    // utilize L2 better.
    GM_ADDR const padded_rowwise_scan = workspace;

    GM_ADDR const intermediate_res =
        IsInclusive ? output_vec : padded_rowwise_scan;
    GM_ADDR const sums =
        IsInclusive ? workspace : padded_rowwise_scan + padded_rowwise_size;

    if ASCEND_IS_AIC {
      tcuscan::KernelRowScan<InputT> op_cube(matmul_size, matmul_size, vec_len);
      op_cube.Init(input_vec, upper_triangular, intermediate_res);
      op_cube.Process();
      PipeBarrier<PIPE_ALL>();
    }

    if ASCEND_IS_AIV {
      tcuscan::KernelReduceTiles<InputT> op_reduce(
          matmul_size * matmul_size / 2, vec_len);
      op_reduce.Init(input_vec, sums);
      op_reduce.Process();
    }

    SyncAll<false /*isAIVOnly*/>();

    if ASCEND_IS_AIV {
      tcuscan::KernelCompleteRows<OutputT, IsInclusive> op(
          matmul_size, matmul_size / 2, vec_len);
      op.Init(intermediate_res, sums, output_vec);
      op.Process(starting_value);
    }
  }
}
