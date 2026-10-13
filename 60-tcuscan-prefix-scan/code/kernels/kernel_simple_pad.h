/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
 *
 * @file kernel_simple_pad.h
 * @brief Kernel implementing a Vector simple pad kernel operation.
 */
#pragma once

#include "ascendc_kernel_operator.h"
#include "tcuscan_utils.h"

using namespace AscendC;

namespace tcuscan {

/**
 * @brief Returns a copy of the last unaligned chunk of the input data.
 *
 */
template <typename T = half>
class KernelSimplePad {
  constexpr static uint32_t BUFFER_NUM = 1;

 public:
  /**
   * @brief Class constructor.
   *
   * @param [in] vec_len Dimension of the input vectors.
   * @param [in] align_len Number of elements that the output will be aligned
   * to.
   */
  __aicore__ inline KernelSimplePad(uint32_t vec_len, uint32_t align_len)
      : vec_len_(vec_len),
        align_len_(align_len),
        tail_len_(vec_len % align_len_),
        pad_len_(align_len_ - tail_len_),
        pad_offset_start_(scalar::AlignDown(vec_len, align_len_)) {}

  /**
   * @brief Initialize global and local memory structures.
   *
   * @param [in] vec_in Pointer to the input vector in global memory.
   * @param [in] vec_out Pointer to the output vector in global memory.
   */
  __aicore__ inline void Init(GM_ADDR vec_in, GM_ADDR vec_out) {
    global_in_.SetGlobalBuffer((__gm__ T*)vec_in, vec_len_);
    global_out_.SetGlobalBuffer((__gm__ T*)vec_out, align_len_);

    pipe.InitBuffer(in_q_, BUFFER_NUM, align_len_ * sizeof(T));
    pipe.InitBuffer(out_q_, BUFFER_NUM, align_len_ * sizeof(T));
  }

  /**
   * @brief Run the kernel.
   *
   */
  __aicore__ inline void Process() {
    if (pad_len_ == 0 or GetBlockIdx() > 0) {
      return;
    }

    copy::CopyGmToVec(in_q_, global_in_[pad_offset_start_], tail_len_);
    ProcessTile();
    copy::CopyVecToGm(global_out_, out_q_, tail_len_);
  }

 private:
  __aicore__ inline void ProcessTile() {
    LocalTensor<T> vec_in_lt = in_q_.DeQue<T>();
    const LocalTensor<T> vec_out_lt = out_q_.AllocTensor<T>();
    DataCopy(vec_out_lt, vec_in_lt, align_len_);
    out_q_.EnQue<T>(vec_out_lt);
    in_q_.FreeTensor<T>(vec_in_lt);
  }

  TPipe pipe;

  TQue<QuePosition::VECIN, BUFFER_NUM> in_q_;
  TQue<QuePosition::VECOUT, BUFFER_NUM> out_q_;

  GlobalTensor<T> global_in_;
  GlobalTensor<T> global_out_;

  const uint32_t vec_len_;
  const uint32_t align_len_;
  const uint32_t tail_len_;
  const uint32_t pad_len_;
  const uint32_t pad_offset_start_;
};

/**
 * @brief Run the `simple_pad` kernel.
 *
 * @tparam ForceMixMode Indicates if kernel should schedule dummy cube
 * operations to make sure it runs in mix mode. Can be safely set to `false`
 * when running inside another mix mode kernel.
 *
 * @param [in] vec_in Pointer to the input vector.
 * @param [in] vec_out Pointer ot the output vector.
 * @param [in] vec_len Dimension of the input vector.
 * @param [in] align_len Length of the alignment.
 */
template <bool ForceMixMode = true, typename InputT>
__aicore__ inline void run_simple_pad(GM_ADDR vec_in, GM_ADDR vec_out,
                                      uint32_t vec_len, uint32_t align_len) {
  if constexpr (ForceMixMode) {
    exec_mode::EnableCubeCores();
  }
  if ASCEND_IS_AIV {
    KernelSimplePad<InputT> op(vec_len, align_len);
    op.Init(vec_in, vec_out);
    op.Process();
  }
}

}  // namespace tcuscan
