/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 *
 * @file kernel_arithmetic_progression.h
 * @brief Kernel generating an arithmetic progression in global memory.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

namespace tcuscan {

using namespace AscendC;

/**
 * @brief Generates an arithmetic sequence in global memory.
 *
 * @tparam T Data type of elements.
 * @tparam FirstVal Value of the first element in the generated sequence.
 * @tparam DiffVal Difference between consecutive elements.
 */
template <typename T, T FirstVal, T DiffVal>
class KernelArithmeticProgression {
  template <typename _T>
  constexpr static bool IS_DT_SUPPORTED =
      std::is_same_v<_T, half> || std::is_same_v<_T, float> ||
      std::is_same_v<_T, int16_t> || std::is_same_v<_T, int32_t>;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] buf_len Number of elements in the buffer.
   * @param [in] tile_size Number of elements processed by each block in a
   * single iteration.
   */
  __aicore__ inline KernelArithmeticProgression(uint32_t buf_len,
                                                uint32_t tile_size)
      : block_num_(GetBlockNum() * GetTaskRation()),
        buf_len_(buf_len),
        tile_size_(tile_size),
        single_iter_tiles_(block_num_),
        single_iter_elems_(tile_size_ * single_iter_tiles_),
        num_tiles_(scalar::CeilDiv(buf_len_, tile_size_)) {
    static_assert(IS_DT_SUPPORTED<T>, "Unsupported data types.");
#ifdef ASCENDC_CPU_DEBUG
    // ArithProgression instruction uses float under the hood, what might
    // cause problems for large int32 numbers. Some cases are unsupported
    // for now.
    if constexpr (std::is_same_v<T, int32_t>) {
      const uint32_t fp32_max_val_unit_prec = 16777216u;
      // Only the first iteration uses ArithProgression.
      const uint32_t max_val_from_arith_prog =
          FirstVal + single_iter_elems_ - 1;
      ASCENDC_ASSERT(max_val_from_arith_prog <= fp32_max_val_unit_prec, {
        KERNEL_LOG(KERNEL_ERROR,
                   "The kernel doesn't support generating that large values.");
      });
    }
#endif
  }

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] buffer Pointer to the buffer in global memory.
   */
  __aicore__ inline void Init(GM_ADDR buffer) {
    global_buffer_.SetGlobalBuffer((__gm__ T*)buffer, buf_len_);
    pipe.InitBuffer(vec_buf_, tile_size_ * sizeof(T));
    pipe.InitBuffer(vec_out_q_, 2, tile_size_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    LocalTensor<T> cnt_lt = vec_buf_.Get<T>();
    const bool is_active =
        num_tiles_ / block_num_ > 0 || GetBlockIdx() < num_tiles_ % block_num_;
    if (!is_active) return;

    const uint32_t start_tile_idx = GetBlockIdx();
    uint32_t global_offset = start_tile_idx * tile_size_;
    GenerateInitialTile(cnt_lt, global_offset);
    StoreTile(cnt_lt, global_offset);

    global_offset += single_iter_elems_;
    while (global_offset < buf_len_) {
      GenerateNextTile(cnt_lt);
      StoreTile(cnt_lt, global_offset);
      global_offset += single_iter_elems_;
    }
  }

 private:
  __aicore__ inline void GenerateInitialTile(const LocalTensor<T>& tile_lt,
                                             uint32_t global_offset) {
    const T start_val = FirstVal + global_offset * DiffVal;
    ArithProgression(tile_lt, start_val, DiffVal, tile_size_);
  }

  __aicore__ inline void GenerateNextTile(const LocalTensor<T>& tile_lt) {
    const int32_t diff_between_block_tiles = single_iter_elems_;
    Adds(tile_lt, tile_lt, diff_between_block_tiles, tile_size_);
  }

  __aicore__ inline void StoreTile(const LocalTensor<T>& tile_lt,
                                   uint32_t global_offset) {
    const LocalTensor<T> dst_lt = vec_out_q_.AllocTensor<T>();
    DataCopy(dst_lt, tile_lt, tile_size_);
    vec_out_q_.EnQue(dst_lt);

    const bool is_tail = global_offset + tile_size_ > buf_len_;
    const uint32_t num_elems_to_store =
        is_tail ? buf_len_ - global_offset : tile_size_;
    copy::CopyVecToGm(global_buffer_[global_offset], vec_out_q_,
                      num_elems_to_store);
  }

  TPipe pipe;

  GlobalTensor<T> global_buffer_;
  TBuf<TPosition::VECCALC> vec_buf_;
  TQue<QuePosition::VECOUT, 2> vec_out_q_;

  const uint32_t block_num_;
  const uint32_t buf_len_;
  const uint32_t tile_size_;
  const uint32_t single_iter_tiles_;
  const uint32_t single_iter_elems_;
  const uint32_t num_tiles_;
};

/**
 * @brief Run the arithmetic progression kernel.
 *
 * @tparam T Data type of elements.
 * @tparam FirstVal Value of the first element in the generated sequence.
 * @tparam DiffVal Difference between consecutive elements.
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] buffer Pointer to a buffer in global memory.
 * @param [in] buffer_len Number of elements in the buffer.
 * @param [in] tile_size Number of elements processed by each block in a
 * single iteration.
 */
template <typename T, T FirstVal, T DiffVal, bool ForceMixMode = true>
__aicore__ inline void run_arithmetic_progression(GM_ADDR buffer,
                                                  uint32_t buffer_len,
                                                  uint32_t tile_size) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelArithmeticProgression<T, FirstVal, DiffVal> op(buffer_len, tile_size);
    op.Init(buffer);
    op.Process();
  }
}

}  // namespace tcuscan
