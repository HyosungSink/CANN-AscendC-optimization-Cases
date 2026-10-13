/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_scan_single_core.h
 * @brief Kernel implementing a single-core scan using cube-vector approach.
 */

#include "ascendc_kernel_operator.h"
#include "kernel_row_scan.h"
#include "kernel_simple_pad.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Transforms an inclusive row-wise scan of a vector into an
 * inclusive scan of it.
 *
 * The algorithm utilizes only a single vector core.
 *
 * The algorithm takes an inclusive row-wise scan (row size being equal to
 * `tile_width`) of an input vector (for example the one produced by
 * `KernelRowScan`). Then the algorithm transforms the row-wise scan into a
 * full inclusive scan by iterating over chunks of size `tile_width` and
 * adding to them the sum of all the previous chunks.
 *
 * @tparam T Data type of the input vector.
 * @tparam SyncBefore If true, synchronize with corresponding AIC before each
 * matrix tile iteration
 */
template <typename T, bool SyncBefore = false>
class KernelCompleteRows {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] tile_width Size of the row used by `KernelRowScan`.
   * @param [in] tile_height Number of rows processed in a single
   * iteration.
   * @param [in] vec_len Number of elements in an input vector.
   */
  __aicore__ inline KernelCompleteRows(uint32_t tile_width,
                                       uint32_t tile_height, uint32_t vec_len)
      : tile_width_(tile_width),
        tile_height_(tile_height),
        vec_len_(vec_len),
        tile_size_(tile_width * tile_height),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] input Pointer to row-wise scan in global memory.
   * @param [in] output Pointer to output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR input, GM_ADDR output) {
    global_input_.SetGlobalBuffer((__gm__ T*)input, vec_len_);
    global_output_.SetGlobalBuffer((__gm__ T*)output, vec_len_);

    pipe.InitBuffer(vecin_q_, 1, tile_size_ * sizeof(T));
    pipe.InitBuffer(vecout_q_, 1, tile_size_ * sizeof(T));
  }

  /**
   * @brief Run the kernel - process all tiles.
   *
   * @param [in] running_sum Starting value added to all entries of the
   * output vector.
   */
  __aicore__ inline void Process(T& running_sum = 0) {
    for (uint32_t idx = 0; idx < num_tiles_; idx++) {
      if constexpr (SyncBefore) {
        sync::SyncGroup<sync::GroupSyncDirection::FULL>();
      }
      if (GetBlockIdx() == 0) {
        running_sum = VecIter(idx, running_sum);
      }
    }
  }

 private:
  __aicore__ inline T VecIter(uint32_t iter_idx, T initial_sum) {
    copy::CopyGmToVec(vecin_q_, global_input_[iter_idx * tile_size_]);
    const T sum = ReduceWithVec(initial_sum);
    StoreVecToGlobal(iter_idx);
    return sum;
  }

  __aicore__ inline T ReduceWithVec(T running_sum) {
    LocalTensor<T> vec_lt = vecin_q_.DeQue<T>();
    const LocalTensor<T> vec_buf = vecout_q_.AllocTensor<T>();

    DataCopy(vec_buf, vec_lt, vec_lt.GetSize());
    vecin_q_.FreeTensor(vec_lt);

    uint32_t offset = 0;
    T accumulation = running_sum;
    for (uint32_t i = 0; i < tile_height_; i++) {
      Adds(vec_buf[offset], vec_buf[offset], accumulation, tile_width_);
      offset += tile_width_;
      accumulation = vec_buf.GetValue(offset - 1);
    }

    vecout_q_.EnQue(vec_buf);

    return accumulation;
  }

  __aicore__ inline void StoreVecToGlobal(uint32_t iter_idx) {
    const uint32_t dst_offset = iter_idx * tile_size_;
    const uint32_t tail_len = vec_len_ % tile_size_;
    PipeBarrier<PIPE_ALL>();
    if (iter_idx == num_tiles_ - 1 and tail_len > 0) {
      copy::CopyVecToGm<T>(global_output_[dst_offset], vecout_q_, tail_len);
    } else {
      copy::CopyVecToGm<T>(global_output_[dst_offset], vecout_q_);
    }
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, 1> vecin_q_;
  TQue<QuePosition::VECOUT, 1> vecout_q_;

  GlobalTensor<T> global_input_;
  GlobalTensor<T> global_output_;

  const uint32_t tile_width_;
  const uint32_t tile_height_;
  const uint32_t vec_len_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
};

namespace sc_scan {

/**
 * @brief Calculate the workspace size for single core scan.
 *
 * @tparam T Input data type.
 *
 * @param [in] matmul_size Size of the matmul used in scan.
 * @return Size of the workspace in bytes.
 */
template <typename T>
__aicore__ inline uint32_t get_workspace_size(uint32_t matmul_size) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<T>;
  const uint32_t total_size = matmul_size * matmul_size * sizeof(OutputT);
  return total_size;
}

}  // namespace sc_scan

template <typename InputT>
__aicore__ inline void _run_scan_sc(
    GM_ADDR input_vec, GM_ADDR upper_triangular, GM_ADDR output_vec,
    uint32_t matmul_size, uint32_t vec_aic_len, uint32_t vec_aiv_len,
    tcuscan::cube_unit::CubeOutType_t<InputT>& starting_sum) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  if ASCEND_IS_AIC {
    KernelRowScan<InputT, /* SyncAfter */ true> op_cube(
        matmul_size, matmul_size, vec_aic_len);
    op_cube.Init(input_vec, upper_triangular, output_vec);
    op_cube.Process();
  }

  if ASCEND_IS_AIV {
    KernelCompleteRows<OutputT, /* SyncBefore */ true> op_vec(
        matmul_size, matmul_size, vec_aiv_len);
    op_vec.Init(output_vec, output_vec);
    op_vec.Process(starting_sum);
  }
}

/**
 * @brief Run the single core scan kernel.
 *
 * @tparam InputT Input data type.
 *
 * @param [in] input_vec Pointer to an input vector.
 * @param [in] upper_triangular Pointer to an upper-triangular matrix filled
 * with ones of size \f$\textit{matmul_size} \times \textit{matmul_size}\f$.
 * @param [in] output_vec Pointer to an output vector.
 * @param [in] vec_len Number of elements to process.
 * @param [in] matmul_size Size of the matmul tile used in the kernel.
 * @param [in] workspace Pointer to the kernel workspace.
 * @param [in] starting_sum  starting sum, 0 by default.
 */

template <typename InputT>
__aicore__ inline void run_scan_single_core(
    GM_ADDR input_vec, GM_ADDR upper_triangular, GM_ADDR output_vec,
    uint32_t vec_len, uint32_t matmul_size, GM_ADDR workspace,
    typename tcuscan::cube_unit::CubeOutType_t<InputT> starting_sum = 0) {
  using OutputT = tcuscan::cube_unit::CubeOutType_t<InputT>;

  const uint32_t alignment = matmul_size * matmul_size;
  const uint32_t aligned_vec_len = scalar::AlignDown(vec_len, alignment);
  if ASCEND_IS_AIV {
    // Copy last remaining tile in workspace and dummy-pad it up to
    // `alignment` elements.
    if (aligned_vec_len < vec_len) {
      KernelSimplePad<InputT> op_pad(vec_len, alignment);
      op_pad.Init(input_vec, workspace);
      op_pad.Process();
    }
  }

  PipeBarrier<PIPE_ALL>();
  sync::SyncGroup<sync::GroupSyncDirection::FULL>();

  if (aligned_vec_len > 0) {
    _run_scan_sc<InputT>(input_vec, upper_triangular, output_vec, matmul_size,
                         aligned_vec_len, aligned_vec_len, starting_sum);
  }

  // // Run the kernel on the last matrix ile
  if (aligned_vec_len < vec_len) {
    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    PipeBarrier<PIPE_ALL>();
    if ASCEND_IS_AIC {
      KernelRowScan<InputT, /* SyncAfter */ true> op_cube(
          matmul_size, matmul_size, alignment);
      op_cube.Init(workspace, upper_triangular, workspace);
      op_cube.Process();
    }

    sync::SyncGroup<sync::GroupSyncDirection::FULL>();
    PipeBarrier<PIPE_ALL>();

    if ASCEND_IS_AIV {
      const uint32_t offset = aligned_vec_len * sizeof(OutputT);
      const uint32_t vec_aiv_len = vec_len % alignment;

      KernelCompleteRows<OutputT, /* SyncBefore*/ true> op_vec(
          matmul_size, matmul_size, vec_aiv_len);
      op_vec.Init(workspace, output_vec + offset);
      op_vec.Process(starting_sum);
    }
  }
}

}  // namespace tcuscan
