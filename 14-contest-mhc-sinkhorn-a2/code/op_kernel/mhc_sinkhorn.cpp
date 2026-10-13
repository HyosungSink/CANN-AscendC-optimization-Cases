// Kernel: mHC-Sinkhorn (Sinkhorn-Knopp double stochastic), vectorized step-7
//   = step-6b verbatim (md5 668c24d7, platform Pass 5/5 twice-in-a-row chain §11.21),
//   with ONE variable moved: the fp32 n<N_MAX GM<->UB path stops issuing one
//   DataCopyPad per matrix row and instead strides over the g matrices of the pass.
//
// Purpose: §11.20 #4 localised the last remaining heap that is not the V pipe. For
//   fp32 with n<8 a pass currently costs g*n loads + g*n stores (1024x6 -> 156+156
//   small DMA calls) and that is measurably 15.5us of the 40.7us there (38%), 11.3us
//   of 36.5us at n=4, but only 4.9us of 28.2us at n=8. Striding over w collapses it
//   to n instructions per direction (6), which is a 26x cut on that heap.
//
// The gap units, settled on device this round (three failed builds pinned it):
//   the unit follows the MEMORY SPACE, not the field position.
//     GM side gap = bytes,  burst advances by blockLen + gap;
//     UB side gap = 32B blocks, and the block blockLen (16/24B) lives in is already
//                      counted, so the value is (pitch/32B - 1) = N_MAX - 1 here.
//   First build used bytes on the UB side (240/232). Both n=4 and n=6 failed the fp32
//   element compare; striding only the store changed n=4's deviation (0.25 -> 0.125),
//   so both directions were wrong rather than one side masking the other. The block
//   reading is also what CANN's own two call sites imply:
//   mhc_pre_sinkhorn_premix_base.h:710-711 (srcStride*sizeof(T) vs dstStride/elemInBlock)
//   and attention_worker_combine_split_k.h:161 vs :261 (GM->UB vs UB->GM).
//   code1.md §14.9's "both sides in bytes" is therefore true of the GM side only --
//   worth carrying back to that doc.
//
// pp stays isPad=false, matching step-6b's per-row path: the lanes past row_bytes
//   inside a block keep Seed's NEG_PAD, which Exp flushes to 0. rightPadding was tried
//   here first and made no difference to the verdict, so it is not on the critical path.
//
// Everything else is untouched: G_MAX=31, DMA_WIN=8, the whole V path including its
//   8-row window geometry (Seed still covers rows [n,8) and columns are still folded
//   by N_MAX), and the fp16 branches. So the delta reads off mte2/mte3 + scalar only,
//   and aiv_vec_time is the control that must not move.
//
// 910B3 + CANN 9.0.0, arch22. Layout invariant: a matrix lives in one UB *window* of
// N_MAX=8 rows x RS=8 floats = WIN=64 floats = 256B = 8 blocks, 32B aligned for every
// n in {4,6,8}. Windows are contiguous, so window w starts at element 64*w = block 8*w.
// Every step-4 pattern is identical across windows, which is exactly what
// dstRepStride=8 blocks expresses: the same block gather replays window after window.
//
// Reductions follow the same rule:
//   - row max / row sum: WholeReduce*(dst, src, mask=n, repeatTime=8*g, dstRepStride=1
//     element, srcBlkStride=1, srcRepStride=1 block) -> red[8*w + i] = row i of window
//     w, i.e. one dense block of 8 scalars per matrix;
//   - Brcb(repeatTime=g, {dstBlkStride=1, dstRepStride=8}) expands that block back into
//     a full window (same {1,8} pairing CANN's own MLA kernel uses);
//   - column sums still fold across blocks (no 2201 reduce crosses blocks) in the
//     log2(8) pairwise tree, now with repeatTime=g.
//
// Padding stays a fixed point: rows [n,8) of a window are seeded 0 once per matrix
// (0/(s+eps)==0) and lane padding carries NEG_PAD so Exp flushes it to 0.
//
// GM<->UB: when the fp32 matrix is dense (n == N_MAX) a whole pass is contiguous and
// goes out in bursts of DMA_WIN windows; otherwise rows are copied one DataCopyPad at
// a time, exactly as step-4 did. No strided bursts, so no reliance on gap-vs-start
// stride semantics on this arch.
//
// Only in-core PipeBarrier is used (never SyncAll: it is a cross-core barrier and
// deadlocks whenever batch is not a multiple of coreNum).
#include "kernel_operator.h"

#include "mhc_sinkhorn_tiling.h"
#include "tiling_key_mhc_sinkhorn.h"

#include <type_traits>

using namespace AscendC;

template <typename DT_LOGITS>
class KernelMhcSinkhorn {
public:
    static constexpr int32_t N_MAX = 8;
    static constexpr int32_t RS = 8;    // UB row stride, floats -> exactly 32B
    static constexpr int32_t RH = 16;   // UB row stride, halfs    -> exactly 32B
    static constexpr int32_t WIN = N_MAX * RS;   // one matrix window = 64 floats
    static constexpr int32_t G_MAX = 31;         // matrices per vector pass, hard cap
    // Max windows in one dense DataCopyPad burst. 8 keeps every burst at the 2048B
    // step-5 already ran on device, so the G_MAX change is read off the V pipe alone.
    static constexpr int32_t DMA_WIN = 8;
    // Padding sentinel: far below any plausible logit, far enough from the fp32
    // limit that (sentinel - rowmax) cannot overflow.
    static constexpr float NEG_PAD = -1e30f;

    __aicore__ inline KernelMhcSinkhorn() {}

    __aicore__ inline void Init(GM_ADDR logits, GM_ADDR weights,
                                const MhcSinkhornTilingData &tiling) {
        n_ = tiling.n;
        num_iters_ = tiling.numIters;
        eps_ = tiling.eps;

        const int64_t total = static_cast<int64_t>(tiling.batch) * n_ * n_;
        x_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(logits), total);
        o_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ DT_LOGITS *>(weights), total);

        const uint32_t per_core = (tiling.batchPerCore == 0u) ? 1u : tiling.batchPerCore;
        const uint32_t idx = GetBlockIdx();
        m_begin_ = idx * per_core;
        m_end_ = m_begin_ + per_core;
        if (m_end_ > tiling.batch) m_end_ = tiling.batch;
        if (m_begin_ > tiling.batch) m_begin_ = tiling.batch;

        const uint32_t nf = G_MAX * WIN * sizeof(float);
        pipe_.InitBuffer(mat_buf_, nf);
        pipe_.InitBuffer(red_buf_, nf);
        pipe_.InitBuffer(bcast_buf_, nf);
        pipe_.InitBuffer(cs_buf_, nf);
        if constexpr (!std::is_same<DT_LOGITS, float>::value) {
            const uint32_t nh = G_MAX * N_MAX * RH * sizeof(DT_LOGITS);
            pipe_.InitBuffer(hin_buf_, nh);
            pipe_.InitBuffer(hout_buf_, nh);
        }
    }

    __aicore__ inline void Process() {
        for (uint32_t m = m_begin_; m < m_end_; m += G_MAX) {
            const int32_t g = static_cast<int32_t>(
                (m + G_MAX <= m_end_) ? G_MAX : (m_end_ - m));
            ProcessPass(m, g);
        }
    }

private:
    // Block strides are 1 (consecutive blocks of a window); repeat strides of N_MAX move
    // from one window to the next. cs/red are instead dense (one block per matrix).
    __aicore__ inline BinaryRepeatParams WinBinary() const {
        BinaryRepeatParams rp;
        rp.dstBlkStride = 1;
        rp.src0BlkStride = 1;
        rp.src1BlkStride = 1;
        rp.dstRepStride = N_MAX;
        rp.src0RepStride = N_MAX;
        rp.src1RepStride = N_MAX;
        return rp;
    }

    __aicore__ inline UnaryRepeatParams DenseUnary() const {
        return UnaryRepeatParams(1, 1, 1, 1);
    }

    __aicore__ inline UnaryRepeatParams WinUnary() const {
        return UnaryRepeatParams(1, 1, N_MAX, N_MAX);
    }

    __aicore__ inline void Seed(LocalTensor<float> mat, int32_t n, int32_t g) {
        for (int32_t w = 0; w < g; ++w) {
            Duplicate(mat[w * WIN], NEG_PAD, WIN);
        }
        // The column tree always folds N_MAX rows, so rows [n, N_MAX) must be additive
        // identities. They sit outside every other write, so seeding once per matrix
        // is enough.
        if (n < N_MAX) {
            for (int32_t w = 0; w < g; ++w) {
                Duplicate(mat[w * WIN + n * RS], 0.0f, (N_MAX - n) * RS);
            }
        }
    }

    __aicore__ inline void LoadRows(int32_t n, int32_t g, int64_t base, LocalTensor<float> mat) {
        Seed(mat, n, g);
        const uint32_t row_bytes = static_cast<uint32_t>(n * sizeof(DT_LOGITS));
        DataCopyExtParams cp{1, row_bytes, 0, 0, 0};
        DataCopyPadExtParams<DT_LOGITS> pp{false, 0, 0, static_cast<DT_LOGITS>(0)};
        if constexpr (std::is_same<DT_LOGITS, float>::value) {
            if (n == N_MAX) {   // dense window: contiguous, in step-5-sized bursts
                for (int32_t w = 0; w < g; w += DMA_WIN) {
                    const int32_t cnt = (w + DMA_WIN <= g) ? DMA_WIN : (g - w);
                    DataCopyExtParams all{1, static_cast<uint32_t>(cnt * WIN * sizeof(float)), 0, 0, 0};
                    DataCopyPad(mat[w * WIN], x_gm_[base + static_cast<int64_t>(w) * WIN], all, pp);
                }
            } else {
                // One strided burst per row index instead of one per matrix row: the g
                // copies of row i are n*n floats apart in GM and WIN floats apart in UB.
                // Unit split, measured this round + pinned by CANN's own CopyIn
                // (dstStride = (nBurst-1)*align, i.e. stride minus one block):
                //   GM gap  = bytes, and the burst advances by blockLen + gap;
                //   UB  gap = 32B blocks consumed *in addition to* the one block that
                //             blockLen (16/24B < 32B) already occupies -> N_MAX - 1.
                for (int32_t i = 0; i < n; ++i) {
                    DataCopyExtParams sp{static_cast<uint16_t>(g), row_bytes,
                                         static_cast<uint32_t>((n * n - n) * sizeof(DT_LOGITS)),
                                         static_cast<uint32_t>(N_MAX - 1), 0};
                    DataCopyPad(mat[i * RS], x_gm_[base + static_cast<int64_t>(i) * n], sp, pp);
                }
            }
            PipeBarrier<PIPE_ALL>();
        } else {
            LocalTensor<DT_LOGITS> hin = hin_buf_.Get<DT_LOGITS>();
            for (int32_t w = 0; w < g; ++w) {
                for (int32_t i = 0; i < n; ++i) {
                    DataCopyPad(hin[w * N_MAX * RH + i * RH],
                                x_gm_[base + static_cast<int64_t>(w * n + i) * n], cp, pp);
                }
            }
            PipeBarrier<PIPE_ALL>();
            for (int32_t w = 0; w < g; ++w) {
                for (int32_t i = 0; i < n; ++i) {   // rows [n,N_MAX) keep their 0 seed
                    Cast(mat[w * WIN + i * RS], hin[w * N_MAX * RH + i * RH],
                         RoundMode::CAST_NONE, n);
                }
            }
            PipeBarrier<PIPE_ALL>();
        }
    }

    __aicore__ inline void StoreRows(int32_t n, int32_t g, int64_t base, LocalTensor<float> mat) {
        const uint32_t row_bytes = static_cast<uint32_t>(n * sizeof(DT_LOGITS));
        DataCopyExtParams wo{1, row_bytes, 0, 0, 0};
        if constexpr (std::is_same<DT_LOGITS, half>::value) {
            LocalTensor<DT_LOGITS> hout = hout_buf_.Get<DT_LOGITS>();
            for (int32_t w = 0; w < g; ++w) {
                for (int32_t i = 0; i < n; ++i) {   // only live rows are written out
                    Cast(hout[w * N_MAX * RH + i * RH], mat[w * WIN + i * RS],
                         RoundMode::CAST_RINT, n);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();   // VEC -> MTE1: last Div/Cast must land before DMA
        if constexpr (std::is_same<DT_LOGITS, float>::value) {
            if (n == N_MAX) {
                for (int32_t w = 0; w < g; w += DMA_WIN) {
                    const int32_t cnt = (w + DMA_WIN <= g) ? DMA_WIN : (g - w);
                    DataCopyExtParams all{1, static_cast<uint32_t>(cnt * WIN * sizeof(float)), 0, 0, 0};
                    DataCopyPad(o_gm_[base + static_cast<int64_t>(w) * WIN], mat[w * WIN], all);
                }
            } else {
                // Mirror of the load: the unit follows the memory space, not the field
                // (src here is UB -> 32B blocks, dst is GM -> bytes). Same split CANN uses
                // at attention_worker_combine_split_k.h:261 for a UB->GM strided burst.
                // blockLen caps what reaches GM, so the lanes past row_bytes inside each
                // source block are simply not written.
                for (int32_t i = 0; i < n; ++i) {
                    DataCopyExtParams sp{static_cast<uint16_t>(g), row_bytes,
                                         static_cast<uint32_t>(N_MAX - 1),
                                         static_cast<uint32_t>((n * n - n) * sizeof(DT_LOGITS)), 0};
                    DataCopyPad(o_gm_[base + static_cast<int64_t>(i) * n], mat[i * RS], sp);
                }
            }
        } else {
            LocalTensor<DT_LOGITS> hout = hout_buf_.Get<DT_LOGITS>();
            for (int32_t w = 0; w < g; ++w) {
                for (int32_t i = 0; i < n; ++i) {
                    DataCopyPad(o_gm_[base + static_cast<int64_t>(w * n + i) * n],
                                hout[w * N_MAX * RH + i * RH], wo);
                }
            }
        }
        PipeBarrier<PIPE_ALL>();   // the next pass's MTE2 reuses these UB blocks
    }

    // red[8*w + i] = max of row i of window w; one Brcb expands it per window.
    __aicore__ inline void SubtractRowMax(int32_t n, int32_t g, LocalTensor<float> mat,
                                          LocalTensor<float> red, LocalTensor<float> bcast) {
        WholeReduceMax<float>(red, mat, n, N_MAX * g, 1, 1, 1, ReduceOrder::ORDER_ONLY_VALUE);
        PipeBarrier<PIPE_V>();
        Brcb<float>(bcast, red, g, BrcbRepeatParams(1, N_MAX));
        PipeBarrier<PIPE_V>();
        Sub<float>(mat, mat, bcast, static_cast<uint64_t>(n * RS), g, WinBinary());
    }

    __aicore__ inline void RowNormalize(int32_t n, int32_t g, LocalTensor<float> mat,
                                        LocalTensor<float> red, LocalTensor<float> bcast) {
        WholeReduceSum<float>(red, mat, n, N_MAX * g, 1, 1, 1);
        PipeBarrier<PIPE_V>();
        Adds<float>(red, red, eps_, static_cast<uint64_t>(n), g, DenseUnary());
        Brcb<float>(bcast, red, g, BrcbRepeatParams(1, N_MAX));
        PipeBarrier<PIPE_V>();
        Div<float>(mat, mat, bcast, static_cast<uint64_t>(n * RS), g, WinBinary());
    }

    // Column sums accumulate across blocks (rows live one per 32B block) and no 2201
    // reduce instruction crosses blocks, so this is step-4's pairwise tree, replayed
    // over g windows by one instruction per level. Scratch is bcast, dead once the row
    // Div has consumed it. cs is dense: one block of N_MAX column sums per window.
    __aicore__ inline void ColNormalize(int32_t n, int32_t g, LocalTensor<float> mat,
                                        LocalTensor<float> cs, LocalTensor<float> scratch) {
        Add<float>(scratch, mat, mat[(N_MAX / 2) * RS],
                   static_cast<uint64_t>((N_MAX / 2) * RS), g, WinBinary());
        Add<float>(scratch, scratch, scratch[(N_MAX / 4) * RS],
                   static_cast<uint64_t>((N_MAX / 4) * RS), g, WinBinary());
        BinaryRepeatParams fold = WinBinary();
        fold.dstRepStride = 1;            // cs advances one block per window
        Add<float>(cs, scratch, scratch[(N_MAX / 8) * RS], static_cast<uint64_t>(RS), g, fold);
        PipeBarrier<PIPE_V>();
        Adds<float>(cs, cs, eps_, static_cast<uint64_t>(RS), g, DenseUnary());
        // The divisor block is held still while dst/src0 step block by block.
        BinaryRepeatParams rp = WinBinary();
        rp.src1BlkStride = 0;
        rp.src1RepStride = 1;
        Div<float>(mat, mat, cs, static_cast<uint64_t>(n * RS), g, rp);
    }

    __aicore__ inline void ProcessPass(int64_t m, int32_t g) {
        const int32_t n = static_cast<int32_t>(n_);
        const int64_t base = m * n * n;

        LocalTensor<float> mat = mat_buf_.Get<float>();
        LocalTensor<float> red = red_buf_.Get<float>();
        LocalTensor<float> bcast = bcast_buf_.Get<float>();
        LocalTensor<float> cs = cs_buf_.Get<float>();

        LoadRows(n, g, base, mat);
        SubtractRowMax(n, g, mat, red, bcast);
        Exp<float>(mat, mat, static_cast<uint64_t>(n * RS), g, WinUnary());
        RowNormalize(n, g, mat, red, bcast);
        ColNormalize(n, g, mat, cs, bcast);

        for (uint32_t it = 1; it < num_iters_; ++it) {
            RowNormalize(n, g, mat, red, bcast);
            ColNormalize(n, g, mat, cs, bcast);
        }

        StoreRows(n, g, base, mat);
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> mat_buf_;
    TBuf<TPosition::VECCALC> red_buf_;
    TBuf<TPosition::VECCALC> bcast_buf_;
    TBuf<TPosition::VECCALC> cs_buf_;
    TBuf<TPosition::VECCALC> hin_buf_;
    TBuf<TPosition::VECCALC> hout_buf_;
    GlobalTensor<DT_LOGITS> x_gm_;
    GlobalTensor<DT_LOGITS> o_gm_;
    uint32_t n_;
    uint32_t num_iters_;
    float eps_;
    uint32_t m_begin_;
    uint32_t m_end_;
};

template <typename DT_LOGITS>
__global__ __aicore__ void mhc_sinkhorn(GM_ADDR logits, GM_ADDR mask, GM_ADDR weights,
                                        GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MhcSinkhornTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcSinkhornTilingData, tiling_data, tiling);
    KernelMhcSinkhorn<DT_LOGITS> op;
    op.Init(logits, weights, tiling_data);
    op.Process();
}

template __aicore__ void mhc_sinkhorn<half>(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
template __aicore__ void mhc_sinkhorn<float>(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR);
