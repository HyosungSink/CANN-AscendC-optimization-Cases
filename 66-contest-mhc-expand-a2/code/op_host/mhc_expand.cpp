// Host侧Tiling实现
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include <algorithm>
#include <cstdint>

#include "../op_kernel/mhc_expand_tiling.h"
#include "../op_kernel/tiling_key_mhc_expand.h"

namespace {
// 切分模式枚举（与 TilingData.splitMode 一致）
constexpr uint32_t SPLIT_ROW = 0;          // 按 token 行切核（首选）
constexpr uint32_t SPLIT_ROW_STREAM = 1;   // 前向：把 (s, k) 展开为 S*m 个写任务
constexpr uint32_t SPLIT_ELEMENT = 2;      // 极端小 shape：按 (s, jt) 扁平切
}  // namespace

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        // ---- 平台信息 ----
        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        uint64_t ub_size = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub_size);
        if (ub_size == 0) {
            ub_size = 192 * 1024;   // 防御：默认 910B UB 约 192KB
        }

        // ---- 输入信息 ----
        const gert::Tensor *tensor_x = context->GetRequiredInputTensor(0);
        ge::DataType dtype_x = tensor_x->GetDataType();
        int dtype_size_x = ge::GetSizeByDataType(dtype_x);
        const gert::StorageShape *in_storage_shape = context->GetInputShape(0);
        const gert::Shape &in_shape = in_storage_shape->GetStorageShape();
        size_t in_rank = in_shape.GetDimNum();

        // ---- 算子属性 ----
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_mhc_mult = attrs->GetInt(0);
        const bool *attr_backward = attrs->GetBool(1);
        int64_t m = (attr_mhc_mult != nullptr) ? *attr_mhc_mult : 2;
        bool backward = (attr_backward != nullptr) ? *attr_backward : false;

        if (m < 1) {
            return ge::GRAPH_FAILED;   // 扩展倍数必须 >= 1
        }

        // ---- 语义消歧：按 backward + rank 解析 S / D ----
        uint32_t S = 0, D = 0;
        if (!backward) {
            if (in_rank != 2) return ge::GRAPH_FAILED;   // 前向期望 [S, D]
            S = static_cast<uint32_t>(in_shape.GetDim(0));
            D = static_cast<uint32_t>(in_shape.GetDim(1));
        } else {
            if (in_rank != 3) return ge::GRAPH_FAILED;   // 反向期望 [S, m, D]
            if (static_cast<int64_t>(in_shape.GetDim(1)) != m) return ge::GRAPH_FAILED;
            S = static_cast<uint32_t>(in_shape.GetDim(0));
            D = static_cast<uint32_t>(in_shape.GetDim(2));
        }
        if (S == 0 || D == 0) return ge::GRAPH_FAILED;

        // ---- 配置 tiling key（dtype 模板域：fp16 / bf16；backward 布尔模板域） ----
        uint32_t DT_X = 0;
        if (dtype_x == ge::DT_FLOAT16) {
            DT_X = C_DT_FLOAT16;
        } else if (dtype_x == ge::DT_BF16) {
            DT_X = C_DT_BF16;
        } else {
            return ge::GRAPH_FAILED;
        }
        uint32_t BACKWARD = backward ? 1u : 0u;
        ASCENDC_TPL_SEL_PARAM(context, DT_X, BACKWARD);

        // ---- 填充 TilingData ----
        MhcExpandTilingData *tiling = context->GetTilingData<MhcExpandTilingData>();
        tiling->S = S;
        tiling->D = D;
        tiling->m = static_cast<uint32_t>(m);
        tiling->backward = backward ? 1u : 0u;

        // D 方向大 tile：32B 对齐（fp16/bf16 = 16 元素），典型 512~2048；
        // 前向攒批路径在 UB 里同时握 2*BS 份 tile 做环（见 op_kernel 的 fwd_b0_~b7_），
        // kernel 侧按"环占用字节"反推 BS：整行且 tile<=6KB 用 8 格、tile>=12KB 用 4 格（R10）。
        // 所以这个 /4 是**环占用的字节上限**：放宽预算等于放弃前向 barrier 批处理，两者必须一起改。
        const uint64_t elem_size = static_cast<uint64_t>(dtype_size_x);
        const uint64_t ub_budget = ub_size / 4;
        uint32_t d_tile_len = 0;
        if (static_cast<uint64_t>(D) * elem_size <= ub_budget) {
            d_tile_len = D;                              // 整行模式：dTileNum = 1
        } else {
            uint64_t t = std::min<uint64_t>(2048, D);
            uint64_t max_t = ub_budget / elem_size;
            t = std::min(t, max_t);
            if (t > 16) t = (t / 16) * 16;               // 32B 对齐
            if (t < 16) t = 16;
            if (t > D) t = D;
            if (t < 1) t = 1;                            // 极端兜底
            d_tile_len = static_cast<uint32_t>(t);
        }
        tiling->dTileLen = d_tile_len;
        tiling->dTileNum = (D + d_tile_len - 1) / d_tile_len;
        tiling->dTailLen = D - (tiling->dTileNum - 1) * d_tile_len;

        // ---- 合批量与合批资格（R15 真机实测 2026-09-22，code1.md §23.4~§23.8b）----
        // 四条硬条件与 kernel 侧 MergeRows() 逐字对齐：前向 + 整行(dTileNum==1) +
        // tile<=6144B(FWD_SMALL_THRESH) + 32B 对齐。第五条件"每核 >=2 行"要由核数决定，
        // 这里用 merge_cap*2<=S 一起判掉 ⇒ merge_ok 为真时按 merge_cap 开核，合批**必然**发生，
        // 不需要到了设备侧再退回来。（下界 8 因此把资格门槛抬到 S>=16；S<16 退回原 ELEMENT 面。）
        const uint64_t io_bytes = static_cast<uint64_t>(m + 1) * static_cast<uint64_t>(S) *
                                  static_cast<uint64_t>(D) * elem_size;
        const uint64_t tile_bytes = static_cast<uint64_t>(d_tile_len) * elem_size;
        // 下界取 8 而不是 4（真机读数，§23.8b）：io=12/24/48KB 合批态 blk=4 与 blk=8 的差
        // <=0.3us、在单点噪声带内（2.1/2.3、2.0/2.0、1.8/2.1），而 S=200/tile=256B 那条
        // （每核行数最大的一段）谷底正落在 blk=8：2.2us，比 io/8192 给的 18 核快 27%。
        // 取 4 的那一版在平台上把唯一出带的一条用例打慢了 38.8% —— 本地既然分辨不出收益，
        // 就退回到两侧都有读数的那个值。blk=2 仍然明确变差（每核两批 => 两道 barrier）。
        uint64_t merge_cap = 8;
        while ((merge_cap + 1) * (merge_cap + 1) <= io_bytes / 4096) ++merge_cap;
        // io ≥ 4.2MB 之后进入带宽饱和段，√ 律会让位给"每核最多 128KiB"这条满开核线：
        // 实测 io=5.24MB 的谷底回到 blk=40（5.05us），按 √ 给的 35 反而贵 3.8%（§23.6c）。
        // 交点 sqrt(io/4096)=io/131072 恰在 4.19MB，与数据一致；io>6.55MB 时两者都 ≥num_aiv
        // ⇒ 被后面的 `core_cap < block_dim` 夹成满开核，中大档天然恒等。
        const uint64_t sat_cap = io_bytes / 131072;
        if (sat_cap > merge_cap) merge_cap = sat_cap;
        const bool merge_ok = !backward && tiling->dTileNum == 1 && tile_bytes <= 6144 &&
                              (tile_bytes % 32) == 0 && merge_cap * 2 <= S;

        // ---- 切分决策（"切分最优"得分点，DESIGN.md §3.5） ----
        uint32_t num_aiv = static_cast<uint32_t>(num_cores_aiv);
        if (num_aiv == 0) num_aiv = 1;
        uint64_t total_tasks = 0;
        uint32_t split_mode = SPLIT_ROW;
        uint32_t block_dim = num_aiv;

        // R15/H4：S<num_aiv 的前向原来一律被推到 STREAM/ELEMENT，于是永远过不了合批的门。
        // merge_ok 已经保证了"改走 ROW 之后每核至少 2 行"，所以这里放它进来是合批生效的
        // 前置条件，不是额外的自由度（实测 S=32/S=8 两条 −24%~−31%，§23.6b）。
        if (S >= num_aiv || merge_ok) {
            // 行数足够 → 按 token 行切核（首选，每核负责连续若干整行）
            split_mode = SPLIT_ROW;
            total_tasks = S;
        } else if (!backward) {
            uint64_t stream_tasks = static_cast<uint64_t>(S) * m;
            if (stream_tasks >= num_aiv) {
                // 前向 S 小：把 (s, k) 展开为 S*m 个写任务，充分用核
                split_mode = SPLIT_ROW_STREAM;
                total_tasks = stream_tasks;
            } else {
                // 前向极端小 shape：按 (s, k, jt) 扁平切
                split_mode = SPLIT_ELEMENT;
                total_tasks = static_cast<uint64_t>(S) * m * tiling->dTileNum;
            }
        } else {
            // 反向小 S：按 (s, jt) 切（每个输出元素由唯一核负责，免跨核归约，无需 workspace）
            split_mode = SPLIT_ELEMENT;
            total_tasks = static_cast<uint64_t>(S) * tiling->dTileNum;
        }

        if (total_tasks < num_aiv) block_dim = static_cast<uint32_t>(total_tasks);
        if (block_dim == 0) block_dim = 1;

        // ---- 小档少开核（真机实测 2026-09-20 §14 定则，2026-09-21 R11 §19 复核并分向）----
        // 每多启动一个 block 要多付一次派发（R11-B 用空 kernel 直接量出这条地板：
        // 固定 1.2us + 每核约 75~90ns），所以"每核分到的字节"低于拐点时开核反而更慢：
        //   fwd-fp16-small(96KB) blk=40 -> 4.5us，blk=16 -> 3.0us，blk=12 -> 2.9us，blk=4 -> 3.9us
        //   bwd-fp16-small(96KB) blk=40 -> 4.8us，blk=24 -> 3.7us，blk=16 -> 3.8us，blk=8 -> 4.3us
        // 前向 IO = S*D + S*m*D、反向同样 = (1+m)*S*D（谁当输入不影响总量），故一条式子覆盖两向。
        // 拐点**分向**取：前向在 R11 把 barrier 按块摊薄（整行小 tile 攒 4 块）之后谷底左移到
        // 8KB/核 = 96KB/12，c0/c2 各三轮交替读到 2.6us（同形 BS=1@16 为 3.0us）；反向不经攒批
        // 那条路，谷底仍在 6KB/核。medium(42MB)/large(8.4GB) 远不到拐点，仍按 num_aiv 满开核。
        // 注意：这条只在 IO 不小于 core_floor*min_io_per_core 的那一段成立，再往小走方向相反（R13）。
        // R15：合批段用上面算好的 sqrt 谷底（下界 8，理由见上面的 §23.8b 注释）；其余形状一律
        // 沿用原来的线性律。上面那条 8KB/核 的拐点是在**不合批**（每核发起数 = tpc*(1+m)）下
        // 量出来的，合批把发起数与每核行数解耦之后谷底左移、且在 4~8 之间是平的：
        // io=96/150/192/192/216KB 五条取 blk=8 时对原线性律(12/18/24/24/27)是
        // -16%/-27%/-31%/-28%/-35%（code1.md §23.4、§23.8b 两张表；blk=4 只再快 <=0.1us，不出噪声带）。
        // ---- R13（真机实测 2026-09-21，code1.md §20）：小 IO 段上面这条要反向，用核数下界封住 ----
        // 固定派发成本在 blk<=8 内几乎不涨（空 kernel 直测 1.2~2.1us），而每核串行 DMA 条数
        // 与核数成反比 ⇒ IO 掉到几十 KB 以下时把核数压到 1~6 是净亏。同机同码 msprof 剔首 mean：
        //   前向 3KB blk1->8 为 3.3->2.2us，12KB 5.0->2.3us，24KB(原 blk=3) 3.7->2.7us
        //   反向 3KB 4.0->2.1us，24KB(原 blk=4) 4.1->3.0us；反向 96KB 要到 blk16 才降到 3.5us
        // 8 是这条阶梯在 3KB~96KB 全段的谷底（一律抬到 16 会回弹 0.3~0.5us）。
        // io_bytes >= 8*每核拐点（前向 8KB*8=64KB、反向 6KB*8=48KB）时 core_cap 本就不小于 8，
        // 这个 max 是恒等变换 ⇒ 中大形状拿到的核数不变。
        // 合批段也吃这条下界：merge_cap 的初值就是 8（上面的 R15 注释给了理由），于是提交 7 的
        // 那句 `if (core_cap < 8)` 在这里**一字未动** ⇒ 本轮与提交 7 的全部差异只剩
        // "合批资格成立时 core_cap 取 merge_cap 而不是 io/8192"与那一行路由，别的一条没有。
        // R16：上面那句"反向谷底仍在 6KB/核"作废，反向改走下面的核数律（§23.11~§23.13）。
        // 前向 io/8192 一支与合批 merge_cap 一支一字未动，且 `if (core_cap < 8)` 那句仍在
        // 链尾原样保留（对 merge_cap≥8 与反向下界 8 都是恒等变换）⇒ 前向全部用例的 block_dim
        // 与提交 9 逐决策相同，可直接当 A/B 的对照组读。
        // 反向合批资格：与 kernel 侧 BwdWideRows 用同一批量、同一条式子（整行模式 ∧ m>=2 ∧
        // 32B 对齐 ∧ cap>=2 ⇔ tile<=3072），⇒ 两边由构造一致。多出来的 S>=32 是"只在我量过的
        // 那段换律"：合批的 blk 阶梯计时只覆盖 S=64 一条形状（p23 v*/p24 v* 两族），
        // S=96/128 只有数值门禁覆盖过 ⇒ S<32 一段一律留在提交 10 那条原律上。
        const bool bwd_wide_ok = backward && split_mode == SPLIT_ROW && tiling->dTileNum == 1 &&
                                 m >= 2 && tile_bytes <= 3072 && (tile_bytes % 32) == 0 && S >= 32;
        uint64_t core_cap;
        if (merge_ok) {
            core_cap = merge_cap;
        } else if (backward) {
            // 反向谷底跟**核数**，既不跟每核字节（io/6144）也不跟每核发起数 tpc。
            // 4 次独立扫描（p6/p6b/p7/p8）、12 条形状 ×8 个 blk 臂的真机阶梯：
            //   io ≤ 190KiB ∧ S ≥ 32 ⇒ 谷底恒在 blk=16：c24(48KiB) 4.45→3.55 −20.2%、
            //     c22(160KiB) 4.45→4.4、c25/c26/c27 3.1~3.3→3.0~3.2（tpc=2/4/6/8 都拿 16 核）；
            //   S < 32 ⇒ 16 核会开出空核（反向 total_tasks=S*dTileNum，blk>S 的核没有任务），
            //     c30(S=8/12KiB) blk=16 比 blk=8 慢 19% ⇒ 中段下界走 max(8, S/2)；
            //   io ≥ 196KiB ⇒ "每核 16KiB"这条线才重新占优（c23 io=288KiB 两次扫描分别给
            //     24 和 32，单点分不开 ⇒ 取两律交点 12288，不追那个分辨率之外的读数）。
            // 换律后的 12 条：5 条快 3.0%~20.2%、7 条恒等、0 条已知变慢。
            // 合批把每核每行那条链缩到 1/L ⇒ k 变小，谷底 blk* = sqrt(k*S/B) 跟着左移；
            // p23(H5c 宽 VEC)/p24 同构建阶梯把这一族在 c5 上的谷底钉在 blk=8
            //   合批：blk 2/4/8/16/32 = 4.13/3.07/**2.65**/2.95/3.95
            //   原路：blk 2/4/8/16/32 = 12.65/7.18/4.65/**3.77**/4.18
            // 不合批时维持提交 10 那条 16 封顶一字不动。
            const uint64_t bwd_top = bwd_wide_ok ? 8 : 16;
            const uint64_t blk_guard = std::max<uint64_t>(8, std::min<uint64_t>(bwd_top, S / 2));
            core_cap = std::max(io_bytes / 12288, blk_guard);
        } else {
            core_cap = io_bytes / 8192;
        }
        if (core_cap < 8) core_cap = 8;
        if (core_cap < block_dim) block_dim = static_cast<uint32_t>(core_cap);

        tiling->splitMode = split_mode;
        tiling->blockDim = block_dim;
        tiling->rowsPerCore = static_cast<uint32_t>((total_tasks + block_dim - 1) / block_dim);
        // 最后一个核的任务数（uint64 计算避免溢出）
        uint64_t tail = total_tasks - static_cast<uint64_t>(tiling->rowsPerCore) * (block_dim - 1);
        tiling->tailRows = static_cast<uint32_t>(tail);

        // ---- 启动核数与 workspace ----
        context->SetBlockDim(block_dim);
        size_t *current_workspace = context->GetWorkspaceSizes(1);
        current_workspace[0] = 0;   // 本实现不需要 workspace
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        const gert::Shape *in_shape = context->GetInputShape(0);
        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        const int64_t *attr_mhc_mult = attrs->GetInt(0);
        const bool *attr_backward = attrs->GetBool(1);
        int64_t m = (attr_mhc_mult != nullptr) ? *attr_mhc_mult : 2;
        bool backward = (attr_backward != nullptr) ? *attr_backward : false;

        size_t in_rank = in_shape->GetDimNum();
        gert::Shape *out_shape = context->GetOutputShape(0);
        if (!backward) {
            // 前向： [S, D] -> [S, m, D]
            if (in_rank != 2 || m < 1) return GRAPH_FAILED;
            out_shape->SetDimNum(3);
            out_shape->SetDim(0, in_shape->GetDim(0));
            out_shape->SetDim(1, m);
            out_shape->SetDim(2, in_shape->GetDim(1));
        } else {
            // 反向： [S, m, D] -> [S, D]
            if (in_rank != 3 || m < 1) return GRAPH_FAILED;
            if (in_shape->GetDim(1) != m) return GRAPH_FAILED;
            out_shape->SetDimNum(2);
            out_shape->SetDim(0, in_shape->GetDim(0));
            out_shape->SetDim(1, in_shape->GetDim(2));
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        ge::DataType in_dtype = context->GetInputDataType(0);   // 返回值类型是 ge::DataType（值）
        context->SetOutputDataType(0, in_dtype);                // 参数类型也是 ge::DataType（值）
        return GRAPH_SUCCESS;
    }
}  // namespace ge

namespace ops {
    class MhcExpand : public OpDef {
    public:
        explicit MhcExpand(const char *name) : OpDef(name) {
            this->Input("x")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("o")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_BF16})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("mhc_mult").AttrType(OPTIONAL).Int(2);
            this->Attr("backward").AttrType(OPTIONAL).Bool(false);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcExpand);
}  // namespace ops
