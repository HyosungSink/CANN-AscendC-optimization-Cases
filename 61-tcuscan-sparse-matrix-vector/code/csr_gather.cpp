#include "kernels/kernel_csr_gather.h"
#include "kernels/tcuscan_utils.h"
#include "tiling/tiling_csr_gather.h"

extern "C" __global__ __aicore__ void csr_gather_fp16(
    GM_ADDR values_in, GM_ADDR cols_in, GM_ADDR x_in, GM_ADDR z_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CSRGatherTiling t;
  GetTilingData(&t, tiling_gm);

  tcuscan::run_csr_gather<half, false>(values_in, cols_in, x_in, z_out,
                                       t.num_elems, t.num_x_elems, t.tile_len,
                                       t.x_tile_elems_max);
}

extern "C" __global__ __aicore__ void csr_gather_int16(
    GM_ADDR values_in, GM_ADDR cols_in, GM_ADDR x_in, GM_ADDR z_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CSRGatherTiling t;
  GetTilingData(&t, tiling_gm);

  tcuscan::run_csr_gather<int16_t, false>(values_in, cols_in, x_in, z_out,
                                          t.num_elems, t.num_x_elems,
                                          t.tile_len, t.x_tile_elems_max);
}

extern "C" __global__ __aicore__ void csr_gather_fp32(
    GM_ADDR values_in, GM_ADDR cols_in, GM_ADDR x_in, GM_ADDR z_out,
    GM_ADDR workspace, GM_ADDR tiling_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

  (void)workspace;
  tcuscan::CSRGatherTiling t;
  GetTilingData(&t, tiling_gm);

  tcuscan::run_csr_gather<float, false>(values_in, cols_in, x_in, z_out,
                                        t.num_elems, t.num_x_elems, t.tile_len,
                                        t.x_tile_elems_max);
}

/**
 * @brief Call the `csr_gather` kernel for FP16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] values_in Pointer to an input buffer.
 * @param [in] cols_in Pointer to an input buffer.
 * @param [in] x_in Pointer to an input buffer.
 * @param [in] z_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_csr_gather_fp16(uint32_t blockDim, void* stream,
                                       uint8_t* values_in, uint8_t* cols_in,
                                       uint8_t* x_in, uint8_t* z_out,
                                       uint8_t* workspace, uint8_t* tiling_gm) {
  csr_gather_fp16<<<blockDim, nullptr, stream>>>(values_in, cols_in, x_in,
                                                 z_out, workspace, tiling_gm);
}

/**
 * @brief Call the `csr_gather` kernel for INT16 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] values_in Pointer to an input buffer.
 * @param [in] cols_in Pointer to an input buffer.
 * @param [in] x_in Pointer to an input buffer.
 * @param [in] z_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_csr_gather_int16(uint32_t blockDim, void* stream,
                                        uint8_t* values_in, uint8_t* cols_in,
                                        uint8_t* x_in, uint8_t* z_out,
                                        uint8_t* workspace,
                                        uint8_t* tiling_gm) {
  csr_gather_int16<<<blockDim, nullptr, stream>>>(values_in, cols_in, x_in,
                                                  z_out, workspace, tiling_gm);
}

/**
 * @brief Call the `csr_gather` kernel for FP32 data type.
 *
 * @param [in] blockDim Number of blocks for the kernel launch.
 * @param [in] stream NPU stream.
 * @param [in] values_in Pointer to an input buffer.
 * @param [in] cols_in Pointer to an input buffer.
 * @param [in] x_in Pointer to an input buffer.
 * @param [in] z_out Pointer to an output buffer.
 * @param [in] workspace Pointer to workspace.
 * @param [in] tiling_gm Pointer to the tiling buffer.
 */
extern "C" void launch_csr_gather_fp32(uint32_t blockDim, void* stream,
                                       uint8_t* values_in, uint8_t* cols_in,
                                       uint8_t* x_in, uint8_t* z_out,
                                       uint8_t* workspace, uint8_t* tiling_gm) {
  csr_gather_fp32<<<blockDim, nullptr, stream>>>(values_in, cols_in, x_in,
                                                 z_out, workspace, tiling_gm);
}
