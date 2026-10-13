// TilingKey模板定义的头文件
#pragma once

#include "ascendc/host_api/tiling/template_argument.h"

ASCENDC_TPL_ARGS_DECL(MhcSinkhorn,
    ASCENDC_TPL_DATATYPE_DECL(DT_LOGITS, C_DT_FLOAT16, C_DT_FLOAT),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_LOGITS, C_DT_FLOAT16, C_DT_FLOAT),
    ),
);