/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022-2024. All rights reserved.
 *
 * @file kernel_copy.h
 * @brief Kernels implementing a copy operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Copy data between tensors in GM through UB using AscendC queues.
 *
 * Kernel copies the data from GM to UB (VECIN), UB (VECIN) to UB (VECOUT) and
 * finally UB (VECOUT) to GM.
 *
 * @tparam T Data type of the buffer elements.
 * @tparam NumBuffers Number of buffers in queues.
 */
template <typename T, uint16_t NumBuffers>
class KernelCopy {
 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Number of elements.
   * @param [in] tile_size Number of elements processed by each block in a
   * single iteration.
   */
  __aicore__ inline KernelCopy(uint32_t vec_len, uint32_t tile_size)
      : vec_len_(vec_len),
        tile_size_(tile_size),
        num_tiles_(scalar::CeilDiv(vec_len_, tile_size_)) {}
  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] src Pointer to the source buffer in global memory.
   * @param [in] dst Pointer to the destination buffer in global memory.
   */
  __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst) {
    global_src_.SetGlobalBuffer((__gm__ T*)src, vec_len_);
    global_dst_.SetGlobalBuffer((__gm__ T*)dst, vec_len_);
    pipe.InitBuffer(vec_in_q_, NumBuffers, tile_size_ * sizeof(T));
    pipe.InitBuffer(vec_out_q_, NumBuffers, tile_size_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   */
  __aicore__ inline void Process() {
    if (GetBlockIdx() == 0 and GetSubBlockIdx() == 0) {
      return;
    }

    uint32_t global_offset = 0;
    for (uint32_t tile_idx = 0; tile_idx < num_tiles_; tile_idx++) {
      const bool is_full_tile = global_offset + tile_size_ <= vec_len_;
      const uint32_t num_elems_to_process_ =
          is_full_tile ? tile_size_ : vec_len_ - global_offset;

      copy::CopyGmToVec<T>(vec_in_q_, global_src_[global_offset],
                           num_elems_to_process_);
      copy::CopyVecToGm<T>(global_dst_[global_offset], vec_out_q_, vec_in_q_,
                           num_elems_to_process_);
      global_offset += tile_size_;
    }
  }

 private:
  TPipe pipe;

  GlobalTensor<T> global_src_;
  GlobalTensor<T> global_dst_;
  TQue<QuePosition::VECIN, NumBuffers> vec_in_q_;
  TQue<QuePosition::VECOUT, NumBuffers> vec_out_q_;

  const uint32_t vec_len_;
  const uint32_t tile_size_;
  const uint32_t num_tiles_;
};

/**
 * @brief Run the copy kernel.
 *
 * @tparam T Data type
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] src Pointer to the source buffer in global memory.
 * @param [in] dst Pointer to the destination buffer in global memory.
 * @param [in] buffer_len Number of elements in the buffer.
 * @param [in] tile_size Number of elements processed by each block in a
 * single iteration.
 */
template <typename T, bool ForceMixMode = true>
__aicore__ inline void run_copy(GM_ADDR src, GM_ADDR dst, uint32_t buffer_len,
                                uint32_t tile_size) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelCopy<T, 2> op(buffer_len, tile_size);
    op.Init(src, dst);
    op.Process();
  }
}

}  // namespace tcuscan
