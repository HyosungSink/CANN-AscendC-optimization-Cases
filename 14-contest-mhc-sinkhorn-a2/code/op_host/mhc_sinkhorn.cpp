// Host侧Tiling实现（本地复现用重建桩，逻辑与比赛工程一致）
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

#include "../op_kernel/mhc_sinkhorn_tiling.h"
#include "../op_kernel/tiling_key_mhc_sinkhorn.h"

namespace optiling {
    static ge::graphStatus TilingFunc(gert::TilingContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }

        auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int32_t num_cores_aiv = platform.GetCoreNumAiv();
        if (num_cores_aiv <= 0) {
            return ge::GRAPH_FAILED;
        }

        const gert::Tensor *tensor_logits = context->GetRequiredInputTensor(0);
        if (tensor_logits == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const ge::DataType dtype_logits = tensor_logits->GetDataType();

        const auto shape = tensor_logits->GetStorageShape();
        const size_t dim_num = shape.GetDimNum();
        if (dim_num != 3 && dim_num != 4) {
            return ge::GRAPH_FAILED;
        }

        const int64_t n = shape.GetDim(dim_num - 1);
        const int64_t n2 = shape.GetDim(dim_num - 2);
        if (n != n2 || (n != 4 && n != 6 && n != 8)) {
            return ge::GRAPH_FAILED;
        }

        int64_t batch = 1;
        for (size_t i = 0; i + 2 < dim_num; ++i) {
            batch *= shape.GetDim(i);
        }
        if (batch <= 0) {
            return ge::GRAPH_FAILED;
        }

        const gert::RuntimeAttrs *attrs = context->GetAttrs();
        if (attrs == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const int64_t *p_iters = attrs->GetInt(0);
        const float *p_eps = attrs->GetFloat(1);
        const int64_t num_iters = (p_iters != nullptr) ? (*p_iters) : 20;
        const float eps = (p_eps != nullptr) ? (*p_eps) : 1e-6f;
        if (num_iters < 1 || num_iters > 100) {
            return ge::GRAPH_FAILED;
        }

        MhcSinkhornTilingData *tiling = context->GetTilingData<MhcSinkhornTilingData>();
        if (tiling == nullptr) {
            return ge::GRAPH_FAILED;
        }
        tiling->batch = static_cast<uint32_t>(batch);
        tiling->n = static_cast<uint32_t>(n);
        tiling->numIters = static_cast<uint32_t>(num_iters);
        tiling->eps = eps;
        tiling->coreNum = static_cast<uint32_t>(num_cores_aiv);
        tiling->batchPerCore = (static_cast<uint32_t>(batch) +
                                static_cast<uint32_t>(num_cores_aiv) - 1u) /
                               static_cast<uint32_t>(num_cores_aiv);

        uint32_t DT_LOGITS = (dtype_logits == ge::DT_FLOAT) ? C_DT_FLOAT : C_DT_FLOAT16;
        ASCENDC_TPL_SEL_PARAM(context, DT_LOGITS);

        context->SetBlockDim(num_cores_aiv);
        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (currentWorkspace != nullptr) {
            currentWorkspace[0] = 0;
        }
        return ge::GRAPH_SUCCESS;
    }
}  // namespace optiling

namespace ge {
    static graphStatus InferShape(gert::InferShapeContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const gert::Shape *shape_x = context->GetInputShape(0);
        gert::Shape *shape_w = context->GetOutputShape(0);
        if (shape_x == nullptr || shape_w == nullptr) {
            return ge::GRAPH_FAILED;
        }
        const size_t dim_num = shape_x->GetDimNum();
        if (dim_num != 3 && dim_num != 4) {
            return GRAPH_FAILED;
        }
        shape_w->SetDimNum(dim_num);
        for (size_t i = 0; i < dim_num; ++i) {
            shape_w->SetDim(i, shape_x->GetDim(i));
        }
        return GRAPH_SUCCESS;
    }

    static graphStatus InferDataType(gert::InferDataTypeContext *context) {
        if (context == nullptr) {
            return ge::GRAPH_FAILED;
        }
        return context->SetOutputDataType(0, context->GetInputDataType(0));
    }
}  // namespace ge

namespace ops {
    class MhcSinkhorn : public OpDef {
    public:
        explicit MhcSinkhorn(const char *name) : OpDef(name) {
            this->Input("logits")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Input("mask")
                .ParamType(OPTIONAL)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Output("weights")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16, ge::DT_FLOAT})
                .Format({ge::FORMAT_ND, ge::FORMAT_ND});
            this->Attr("iterations").AttrType(OPTIONAL).Int(20);
            this->Attr("eps").AttrType(OPTIONAL).Float(1e-06);
            this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
            this->AICore()
                .SetTiling(optiling::TilingFunc)
                .AddConfig("ascend910b");
        }
    };
    OP_ADD(MhcSinkhorn);
}  // namespace ops
