/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file stub_ops.h
 * \brief
 */

#ifndef NN_COMMON_STUB_OPS_H
#define NN_COMMON_STUB_OPS_H

#include "graph/operator_reg.h"
#include "graph/operator.h"

namespace ge {
/**
*@brief Input data for other operators. \n

*@par Inputs:
*x: A tensor. \n

*@par Attributes:
*index: Index of the input tensor.The data type must be int32 or int64.
Assume that net has three data nodes, one should be set 0, another should
be set 1, and the left should be set 2. \n

*@par Outputs:
*y: A tensor. \n

*@par Third-party framework compatibility
*Compatible with the Caffe operator Data.
*/
REG_OP(Data)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(index, Int, 0)
    .OP_END_FACTORY_REG(Data)

/**
*@brief Creates a constant tensor from a tensor-like object. This operator is used for inference.
Operator Const has the same definition as operator Constant. \n

*@par Attributes:
*value: Required. The value and type of the resulting tensor, and no restrictions on type. \n

*@par Outputs:
*y: A constant tensor. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Const.
*/
REG_OP(Const)
    .OUTPUT(y, TensorType::ALL())
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Const)

/**
* @brief Extracts a strided slice of a tensor. Roughly speaking, this op
    extracts a slice of size (end-begin)/stride from the given input tensor.
    Starting at the location specified by begin the slice continues by
    adding stride to the index until all dimensions are not less than end.

* @par Inputs:
* Four inputs, including:
* @li x: A tensor. Must be one of the BasicType: complex128, complex64,
  double, float32, float16, int16, int32, int64, int8, qint16, qint32, qint8,
  quint16, quint8, uint16, uint32, uint64, uint8, bfloat16, complex32,
  hifloat8, float8_e5m2, float8_e4m3fn.Supported format list ["ND"].

* @li begin: A tensor of IndexNumberType: int32 or int64,
  for the index of the first value to select.Supported format list ["ND"].

* @li end: A tensor of IndexNumberType: int32 or int64,
  for the index of the last value to select.Supported format list ["ND"].

* @li strides: A tensor of IndexNumberType: int32 or int64,
  for the increment.Supported format list ["ND"].

* @par Attributes:
* @li begin_mask: A tensor of type int includes all types of int.
      A bitmask where a bit "i" being "1" means to ignore the begin
      value and instead use the largest interval possible.Default value is 0.
* @li end_mask: A tensor of type int includes all types of int.
      Analogous to "begin_mask".Default value is 0.
* @li ellipsis_mask: A tensor of type int includes all types of int.
      A bitmask where bit "i" being "1" means the "i"th position
      is actually an ellipsis.Default value is 0.
* @li new_axis_mask: A tensor of type int includes all types of int.
      A bitmask where bit "i" being "1" means the "i"th
      specification creates a new shape 1 dimension.Default value is 0.
* @li shrink_axis_mask: A tensor of type int includes all types of int.
      A bitmask where bit "i" implies that the "i"th
      specification should shrink the dimensionality.Default value is 0.

* @par Outputs:
* y: A tensor. Has the same type as "x".Supported format list ["ND"].

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator StridedSlice.
*/
REG_OP(StridedSlice)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(begin, TensorType::IndexNumberType())
    .INPUT(end, TensorType::IndexNumberType())
    .INPUT(strides, TensorType::IndexNumberType())
    .ATTR(begin_mask, Int, 0)
    .ATTR(end_mask, Int, 0)
    .ATTR(ellipsis_mask, Int, 0)
    .ATTR(new_axis_mask, Int, 0)
    .ATTR(shrink_axis_mask, Int, 0)
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(StridedSlice)

/**
*@brief Returns x1 + x2 element-wise. Support broadcasting operations.
*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string.
* @li x2: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string. \n

*@par Outputs:
*y: A ND Tensor. Must be one of the following types: bool, int8, int16, int32, int64, uint8, float64,
*     float16, bfloat16, float32, complex128, complex64, complex32, string.
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Add.
*/
REG_OP(Add)
    .INPUT(x1, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_INT16,
                           DT_INT8, DT_UINT8, DT_DOUBLE, DT_COMPLEX128,
                           DT_COMPLEX64, DT_STRING, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Add)

/**
*@brief Multiply tensor with scale.

*@par Inputs:
*One input, including:
* x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types:
* bfloat16, int32, int16, int64, float16, float32, complex32, complex64.
*
*@par Outputs:
* y: A ND Tensor. Has the same dtype and shape as "x".
*
*@par Attributes:
*@li value: An required attribute. Must be float.
*
*@par Third-party framework compatibility:
* Compatible with the PyTorch operator muls.
*@attention Constraints:
* For parameters of the float32 type, there is no precision loss. For INT32 and INT64 parameters,
* precision loss occurs when the parameter value exceeds 2^24. it is recommended to use Mul.
*/
REG_OP(Muls)
     .INPUT(x, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
     .OUTPUT(y, TensorType({DT_FLOAT, DT_INT16, DT_INT32, DT_INT64, DT_FLOAT16, DT_BF16, DT_COMPLEX32, DT_COMPLEX64}))
     .REQUIRED_ATTR(value, Float)
     .OP_END_FACTORY_REG(Muls)

/**
*@brief Return a tensor with the same shape and contents as input. \n

*@par Inputs:
*x: A tensor. Must be one of the following types: float32、float16、int8、
int16、uint16、uint8、int32、int64、uint32、uint64、bool、double、string、bfloat16. \n

*@par Outputs:
*y: A tensor with the same shape、data type and contents as input. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Identity.
*/
REG_OP(Identity)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(Identity)


/**
* @brief Returns x1 * x2 element-wise.
* y = x1 * x2. Support broadcasting operations.

* @par Inputs:
* @li x1: A ND tensor. Must be one of the following types: bool, float16, float32, bfloat16,
* float64, uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.
* @li x2: A ND tensor. Must be one of the following types: bool, float16, float32, bfloat16,
* float64, uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.
* The shape of x1 and x2 must meet the requirements of the broadcast relationship.

* @par Outputs:
* y: A ND tensor. Must be one of the following types: bool, float16, float32, float64, bfloat16,
* uint8, int8, uint16, int16, int32, int64, complex32, complex64, complex128.

* @attention Constraints:
* "x1" and "x2" have incompatible shapes or types.

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Multiply.
*/
REG_OP(Mul)
    .INPUT(x1, "T1")
    .INPUT(x2, "T2")
    .OUTPUT(y, "T3")
    .DATATYPE(T1, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                              DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                              DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T2, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_UINT8, DT_INT8,
                              DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BF16,
                              DT_COMPLEX64, DT_COMPLEX128, DT_COMPLEX32}))
    .DATATYPE(T3, Promote({"T1", "T2"}))
    .OP_END_FACTORY_REG(Mul)

/**
*@brief Returns x1 - x2 element-wise. Support broadcasting operations.
*@par Inputs:
*Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types: int8, int16, int32, int64, uint8, float64,
*     float16, float32, complex128, complex64, complex32, uint16, bfloat16, bool.
* @li x2: A ND Tensor of the same dtype as "x1". \n

*@par Outputs:
*y: A ND Tensor. Has the same dtype as "x1".
*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Subtract.
*/
REG_OP(Sub)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_DOUBLE, DT_UINT8, DT_INT8,
                           DT_UINT16, DT_INT16, DT_INT32, DT_INT64, DT_BOOL,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Sub)

/**
* @brief Broadcasts an array for a compatible shape.
*  Broadcasting is the process of making arrays to have compatible shapes
*  for arithmetic operations. Two shapes are compatible if for each
*  dimension pair they are either equal or one of them is one. When trying
*  to broadcast a Tensor to a shape, it starts with the trailing dimensions,
*  and works its way forward.
*
* @par Inputs:
* @li x: A tensor, support all dtype include(BasicType, bool, string, hifloat8, float8_e5m2, float8_e4m3fn).
* @li shape: A tensor.
*     A 1D tensor of type int32 or int64, for the shape of the desired output.
*
* @par Outputs:
* y: A tensor. Has the same tensor info of "x".
*
* @par Third-party framework compatibility
* Compatible with the TensorFlow operator BroadcastTo.
*
*/
REG_OP(BroadcastTo)
    .INPUT(x, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType({BasicType(), DT_BOOL, DT_STRING, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(BroadcastTo)


/**
*@brief Reshapes a tensor. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: A tensor. All data types are supported. \n
*@li shape: A tensor. Must be one of the following types: int32, int64. Defines the shape of the output tensor. \n

*@par Attributes:
*@li axis: An optional int32 or int64. The first dimension to reshape. Defaults to "0".
*@li num_axes: An optional int32 or int64. The extent of the reshape. Defaults to "-1". \n

*@par Outputs:
*y: A tensor. The same type as input x. \n

*@attention Constraints:
*This operator cannot be directly called by the acllopExecute API. \n

*@par Third-party framework compatibility
*@li Compatible with the TensorFlow operator Reshape.
*@li Compatible with the Caffe operator Reshape.
*/
REG_OP(Reshape)
    .INPUT(x, TensorType::ALL())
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Reshape)

/**
*@brief Returns a tensor of the same dtype and shape as the input tensor with all elements set to zero.

*@par Inputs:
*x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types: BasicType() and variant.
*
*@par Outputs:
*y: A ND Tensor of the same data type as "x".
*
*@attention Constraints:
* The output has the same shape and type as the input.
*
*@par Third-party framework compatibility
* Compatible with the TensorFlow operator zeros_like.
*/
REG_OP(ZerosLike)
    .INPUT(x, TensorType({BasicType(), DT_VARIANT}))
    .OUTPUT(y, TensorType({BasicType(), DT_VARIANT}))
    .OP_END_FACTORY_REG(ZerosLike)

/**
* @brief Flattens the inputs tensor into a 2D matrix. If input tensor has shape (d_0, d_1,..., d_n),
*        then the output will have shape (d_0 X d_1 ... d_(axis-1), d_axis X d_(axis + 1)...X d_n)\n

* @par Inputs:
* One input:
* x: A multi-dimensional tensor. All data types are supported.

* @par Outputs:
* y: A 2D flattened tensor with the contents of the input tensor, with input dimensions up to axis flattened
* to the outer dimension of the output and remaining input dimensions flattened into the inner dimension of the output.
* Has the same type as "x".

* @par Attributes:
* axis: A optional int32, default value is 1. Indicate up to which input dimensions (exclusive) should be flattened
* to the outer dimension of the output. The value for axis must be in the range [-r, r], where r is the rank of
* the input tensor. Negative value means counting dimensions from the back. When axis = 0, the shape of
* the output tensor is (1, (d_0 X d_1 ... d_n), where the shape of the input tensor is (d_0, d_1, ... d_n).

* @par Third-party framework compatibility
* Compatible with TensorFlow / ONNX operator Flatten.
*/
REG_OP(Flatten)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, Int, 1)
    .OP_END_FACTORY_REG(Flatten)

/**
* @brief Returns locations of nonzero / true values in a tensor. \n

* @par Inputs:
* Including:
* x: A Tensor. Must be one of the following types:
  DT_DOUBLE, DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_QINT8,
  DT_QUINT8, DT_INT16, DT_UINT16, DT_INT32, DT_UINT32, DT_QINT32,
  DT_INT64, DT_UINT64, DT_BOOL, DT_COMPLEX64, DT_COMPLEX128 \n

* @par Outputs:
* y: A Tensor of type DT_INT64. \n

* @attention Constraints:
* Where runs on the Ascend AI CPU, which delivers poor performance.\n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Where.
*/

REG_OP(Where)
    .INPUT(x, TensorType({BasicType(), DT_BOOL}))
    .OUTPUT(y, TensorType({DT_INT64}))
    .OP_END_FACTORY_REG(Where)

/**
*@brief Removes dimensions of size 1 from the shape of a tensor. \n

*@par Inputs:
*x: A tensor. All data types are supported. \n

*@par Attributes:
*axis: An optional list of int32 or int64. Defaults to []. If not specified, squeezes all dimensions of size 1.
If specified, only squeezes the dimensions listed. It is an error to squeeze a dimension that is not 1. \n

*@par Outputs:
*y: A tensor. The same type as input x. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator Squeeze.
*/
REG_OP(Squeeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axis, ListInt, {})
    .OP_END_FACTORY_REG(Squeeze)
/**
* @brief Returns x1/x2 element-wise. Support broadcasting operations.

* @par Inputs:
* Two inputs, including:
* @li x1: A ND Tensor. Must be one of the following types:
*    float16, float32, int32, int8, uint8, float64, int64, uint16, int16,
*    complex32, complex64, complex128, bfloat16, the format can be [NCHW,NHWC,ND].
* @li x2: A ND Tensor. Has the same dtype and format as input "x1". \n

* @par Outputs:
* y: A ND Tensor. Has the same dtype and format as input "x1". \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Div.
*/
REG_OP(Div)
    .INPUT(x1, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .INPUT(x2, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OUTPUT(y, TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_UINT8, DT_INT32,
                           DT_DOUBLE, DT_INT64, DT_UINT16, DT_INT16,
                           DT_COMPLEX64, DT_COMPLEX128, DT_BF16, DT_COMPLEX32}))
    .OP_END_FACTORY_REG(Div)

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*@li x: A tensor.
*@li axis: The dimension index at which to expand. \n

*@par Outputs:
*y: A tensor with the same data as input, with an additional dimension inserted at the index specified by axis. \n

*@par Third-party framework compatibility
*Compatible with the TensorFlow operator ExpandDims.
*/
REG_OP(ExpandDims)
    .INPUT(x, TensorType::ALL())
    .INPUT(axis, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y, TensorType::ALL())
    .OP_END_FACTORY_REG(ExpandDims)

/**
* @brief Computes the output as (shift + scale * x) ^ power .

* @par Inputs:
* x: A tensor of type float16, float32 or bfloat16 . \n

* @par Attributes:
* @li power: Optional. Must be one of the following types: float32. Defaults to 1.0.
* @li scale: Optional. Must be one of the following types: float32. Defaults to 1.0.
* @li shift: Optional. Must be one of the following types: float32. Defaults to 0.0 . \n

* @par Outputs:
* y: A tensor. Has the same type and shape as "x".
* @par Third-party framework compatibility
* Compatible with the Caffe operator Power.
*/

REG_OP(Power)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_BF16}))
    .ATTR(power, Float, 1.0)
    .ATTR(scale, Float, 1.0)
    .ATTR(shift, Float, 0.0)
    .OP_END_FACTORY_REG(Power);

/**
*@brief Inserts a dimension of 1 into a tensor's shape. Only the tensor shape is changed, without changing the data. \n

*@par Inputs:
*x: Original tensor. All data types are supported. \n

*@par Attributes:
*axes: List of ints indicating the dimensions to be inserted. Defaults to []. \n

*@par Outputs:
*y: Reshape tensor with same data as input. The same type as input x. \n

*@par Third-party framework compatibility
*Compatible with the Onnx operator Unsqueeze.
*/

REG_OP(Unsqueeze)
    .INPUT(x, TensorType::ALL())
    .OUTPUT(y, TensorType::ALL())
    .ATTR(axes, ListInt, {})
    .OP_END_FACTORY_REG(Unsqueeze)
    
/**
* @brief Pad a tensor .

* @par Inputs:
* Three inputs, including:
* @li x: A Tensor. Must be one of the following types: bfloat16, float16,
*        float32, double, int32, uint8, int16, int8, complex64, int64, qint8,
*        quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64.
* @li constant_values: A Tensor. Must have the same type as input.
* @li paddings: A Tensor of type int32 or int64 . \n

* @par Outputs:
* y: A Tensor of the same type as "x" . \n

* @par Third-party framework compatibility:
* Compatible with TensorFlow operator Pad.
*/
REG_OP(PadV2)
    .INPUT(x, TensorType::BasicType())
    .INPUT(paddings, TensorType::IndexNumberType())
    .INPUT(constant_values, TensorType::BasicType())
    .OUTPUT(y, TensorType::BasicType())
    .OP_END_FACTORY_REG(PadV2)

/**
* @brief Permutes the dimensions according to perm.
         The returned tensor's dimension i will correspond to the input dimension perm[i].

* @par Inputs:
* Two inputs, including:
* @li x: A Tensor. Must be one of the following types:
* bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32, bool, hifloat8, float8_e5m2,
* float8_e4m3fn, and the maximum dimension should not exceed 8 dimensions,
* and the shape should be consistent with output.
* @li perm: A Tensor of type int32 or int64. A permutation of the dimensions of "x", the value
* should be within the range of [0, number of dimensions for self -1].

* @par Outputs:
* y: A Tensor. Has the same type as "x".

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Transpose.
*/
REG_OP(Transpose)
    .INPUT(x, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
                          DT_QINT16, DT_QUINT16, DT_QINT32, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN}))
    .INPUT(perm, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({DT_BF16, DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT64, DT_INT32,
                          DT_UINT8, DT_UINT16, DT_UINT32, DT_UINT64, DT_INT8, DT_INT16,
                          DT_COMPLEX32, DT_COMPLEX64, DT_COMPLEX128, DT_QINT8, DT_QUINT8,
                          DT_QINT16, DT_QUINT16, DT_QINT32, DT_BOOL, DT_HIFLOAT8, DT_FLOAT8_E5M2,
                          DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Transpose)

/**
* @brief Creates a tensor filled with a scalar value.
* This operation creates a tensor of shape "dims" and fills it with "value".
*
* @par Inputs:
* @li dims: A 1D tensor of types int32 or int64. Represents the shape of the output tensor .
        The size of each dimension must be less than or equal to 8. \n

* @li value: A 0D scalar. Specifies the value to fill the returned tensor.
*    Must be one of the following types:
*    bfloat16, float16, float32, double, int32, uint8, int16, int8, complex64, int64, bool,
*    qint8, quint8, qint32, qint16, quint16, uint16, complex128, uint32, uint64, string.
*
* @par Outputs:
* y: A tensor. Has the same type as "value".
*
* @par Third-party framework compatibility
* @li Compatible with the TensorFlow operator Fill.
* @li Compatible with the Caffe operator Filler.
*
*/
REG_OP(Fill)
    .INPUT(dims, TensorType::IndexNumberType())
    .INPUT(value, "T")
    .OUTPUT(y, "T")
    .DATATYPE(T, TensorType({DT_FLOAT, DT_DOUBLE, DT_INT32, DT_UINT8, DT_INT16,
                              DT_INT8, DT_COMPLEX64, DT_INT64, DT_BOOL, DT_QINT8,
                              DT_QUINT8, DT_QINT32, DT_QINT16, DT_QUINT16, DT_UINT16,
                              DT_COMPLEX128, DT_FLOAT16, DT_BF16, DT_UINT32, DT_UINT64, DT_STRING}))
    .OP_END_FACTORY_REG(Fill)

/**
* @brief Also known as a "fully-connected" layer, computes an inner product
* with a set of learned weights, and (optionally) adds biases.
* @par Inputs:
* Four inputs, including:
* @li x: A Tensor of type float16, int8, int4, bf16.
* @li w: A weight matrix of type float16, int8, int4, float32, bf16.
* @li b: An optional Tensor of type float16, int32, float32, bf16.
* @li offset_w: An optional Tensor of type int8, int4.
* Reserved. Only None Supported. \n

* @par Attributes:
* @li num_output: Required. An int, output neuron number. Reserved.
* @li transpose: A bool, specifying weight whether to transpose input w,
* either "true" or "false". Defaults to "false".
* @li axis: Optional. An int, 1 or 2, specifying which dimension the input
* "K" starts from. Defaults to 1.
* The product of the subsequent dimensions starting form first dimension
* or the second dimension is "K".
* @li offset_x: An optional integer for quantized FullyConnection.
* The negative offset added to the input image for int8 type. Ensure offset_x
* within the effective range of int8 [-128, 127]. Defaults to "0". \n

* @par Outputs:
* y: The result tensor of type float16, int32, float32, bf16. \n

* @par Third-party framework compatibility
* Compatible with the Caffe operator InnerProduct. \n

* @par Quantization supported or not
* Yes
*/
REG_OP(FullyConnection)
    .INPUT(x, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT, DT_BF16}))
    .INPUT(w, TensorType({DT_FLOAT16, DT_INT8, DT_INT4, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(b, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .OPTIONAL_INPUT(offset_w, TensorType({DT_INT8, DT_INT4}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_INT32, DT_FLOAT, DT_BF16}))
    .REQUIRED_ATTR(num_output, Int)
    .ATTR(transpose, Bool, false)
    .ATTR(axis, Int, 1)
    .ATTR(offset_x, Int, 0)
    .OP_END_FACTORY_REG(FullyConnection)

/**
* @brief Returns the maximum of elements across dimensions of a Tensor .

* @par Inputs:
* Two inputs, including:
* @li x: A multi-dimensional Tensor of Must be the type of NumberType. Supported format list ["ND"]
* @li axes: A Scalar of type in IndexNumberType(IndexNumberType includes the
  following types: int32, int64.), specifying the axes information
  of the index with the maximum value. Supported format list ["ND"] \n

* @par Attributes:
* keep_dims: A bool, specifying whether to keep dimensions for the output Tensor.
* Optional and defaults to "false". \n
* noop_with_empty_axes: An optional bool. Defaults to "true" .
* - If true, when axes = [], not reduce.
* - If false, when axes = [], reduce all.
* This attribute is valid only for Ascend950 AI Processors and later products.

* @par Outputs:
* y: A multi-dimensional Tensor, specifying the maximum value of the
  corresponding axis in the tensor.
  Has the same type as "x". (If "keep_dims" is set to "false",
  the output dimensions are reduced by "dimension" compared with that of "x".
  Otherwise, the output has one fewer dimension than "x").Supported format list ["ND"]

* @attention Constraints:
* @li The value range of "axes" is [-dims, dims - 1]. "dims"
  indicates the dimension length of "x".
* @li When converting ONNX to OM, if the axes of the ReduceMax operator is empty,
   it is recommended to use the amax function with dim explicitly set to all axes
   (e.g., dim=[0, 1, 2]) to prevent shape inference errors.

* @par Third-party framework compatibility
* Compatible with TensorFlow operator Max.
*/
REG_OP(ReduceMax)
    .INPUT(x, TensorType::NumberType())
    .INPUT(axes, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::NumberType())
    .ATTR(keep_dims, Bool, false)
    .ATTR(noop_with_empty_axes, Bool, true)
    .OP_END_FACTORY_REG(ReduceMax)

/**
*@brief Cast a tensor form src data type to dst data type.

*@par Inputs:
*One input:
* x:A ND or 5HD tensor. Support 1D~8D. Must be one of the following types: bool, float16, float, int8, int32, uint32, uint8, bfloat16, uint1,
   int64, uint64, int16, uint16, double, complex32, complex64, complex128, qint8, quint8, qint16, quint16, qint32,
   hifloat8, float8_e5m2, float8_e4m3fn, float4_e1m2, float4_e2m1.

*@par Attributes:
*dst_type: A required attribute of type int32, specifying the dst data type.

*@par Outputs:
*y:A ND Tensor with same shape as x, and data type is specified by dst_type.

*@attention Constraints:
* @li In the scenario where the data type is converted from float16 to int16: \n
*     If the input data contains inf, inf is converted into the maximum value of int16. \n
*     If the input data contains -inf, -inf is converted into the minimum value of int16. \n
* @li In the scenarios where the data type is converted from INT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2048, 1920).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from FLOAT32 to INT64 and from FLOAT32 to UINT8: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483583).
* @li Atlas Inference Series Product in the scenarios where the data type is converted from INT64 to FLOAT32: \n
*     It can only guarantee that the input data has no precision errors within the range of (-2147483648, 2147483647).
* @li Ascend 950 AI Processor in the scenario where the data type is converted from INT32 to INT4: \n
*     The last dim of x must be an even number.
*/
REG_OP(Cast)
    .INPUT(x, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                          DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                          DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32, DT_BF16, DT_UINT1,
                          DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                          DT_FLOAT4_E1M2, DT_FLOAT4_E2M1}))
    .OUTPUT(y, TensorType({DT_BOOL, DT_FLOAT16, DT_FLOAT, DT_INT8, DT_INT32, DT_UINT32, DT_UINT8,
                           DT_INT64, DT_UINT64, DT_INT16, DT_UINT16, DT_DOUBLE, DT_COMPLEX64,
                           DT_COMPLEX128, DT_QINT8, DT_QUINT8, DT_QINT16, DT_QUINT16, DT_QINT32,
                           DT_BF16, DT_COMPLEX32, DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN,
                           DT_FLOAT4_E1M2, DT_FLOAT4_E2M1, DT_INT4}))
    .REQUIRED_ATTR(dst_type, Int)
    .OP_END_FACTORY_REG(Cast)

/**
*@brief Returns a tensor of the same shape and type with all elements set to one.
*@par Inputs:
* One input:
* x: An ND or 5HD tensor. support 1D ~ 8D. Must be one of the following types: float16,
* float32, int8, uint8, int16, uint16, int32, int64, complex128, bool, double, bfloat16.
*
*@par Outputs:
* y: A ND Tensor of the same dtype as "x".
*
*@par Third-party framework compatibility
* Compatible with TensorFlow operator OnesLike.
*/
REG_OP(OnesLike)
    .INPUT(x, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8,
                          DT_UINT8, DT_INT16, DI_UINT16, DT_INT32,
                          DT_INT64, DT_COMPLEX128, DT_BOOL, DT_BF16}))
    .OUTPUT(y, TensorType({DT_FLOAT16, DT_FLOAT, DT_DOUBLE, DT_INT8,
                           DT_UINT8, DT_INT16, DI_UINT16, DT_INT32,
                           DT_INT64, DT_COMPLEX128, DT_BOOL, DT_BF16}))
    .OP_END_FACTORY_REG(OnesLike)

/**
* @brief Permutes the dimensions according to perm.
         The returned tensor's dimension i will correspond to the input dimension perm[i]. \n

* @par Inputs:
* x: A Tensor. Must be one of the following types: float16, float32, int8, int16, int32, int64, uint8, uint16, uint32, uint64,
* the maximum dimension should not exceed 8 dimensions, and the shape should be consistent with output. \n

* @par Attributes:
* perm: A permutation of the dimensions of "x", the value
* should be within the range of [number of dimensions for self, number of dimensions for self -1]. \n

* @par Outputs:
* y: A Tensor. Has the same type as "x".
* @par Restrictions:
* Warning: THIS FUNCTION IS DEPRECATED. Please use Transpose instead.
*/
REG_OP(TransposeD)
    .INPUT(x, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                        DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
    .OUTPUT(y, TensorType({DT_INT8, DT_INT16, DT_INT32, DT_INT64, DT_UINT8,
                         DT_UINT16, DT_UINT32, DT_UINT64, DT_FLOAT16, DT_FLOAT}))
    .REQUIRED_ATTR(perm, ListInt)
    .OP_END_FACTORY_REG(TransposeD)

/**
* @brief Extracts a slice from a tensor.
*       This operation extracts a slice of size "size" from a tensor "x"
*       starting at the location specified by "offsets".

* @par Inputs:
* @li x: A Tensor. Must be one of the following types:
* bfloat16, float16, float32, double, int64, int32, uint8, uint16, uint32, uint64, int8,
* int16, complex64, complex128, qint8, quint8, qint16, quint16, qint32, hifloat8, float8_e5m2, float8_e4m3fn.
* @li offsets: A Tensor of type int32 or int64. The starting location for the slice.
* @li size: A Tensor of type int32 or int64. The tensor size for the slice. \n

* @attention Constraints:
* @li 0 <= offset[i] <= offset[i] + size[i] <= x_dim[i] for i in [0,n],
* n is the dimension of the tensor "x". \n
* @li offsets, size and x must have the same rank.

* @par Outputs:
* y: A Tensor. Has the same type as "x". The slice extracted from the tensor. \n

* @par Third-party framework compatibility
* Compatible with the TensorFlow operator Slice.
*/
REG_OP(Slice)
    .INPUT(x, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .INPUT(offsets, TensorType::IndexNumberType())
    .INPUT(size, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType({BasicType(), DT_HIFLOAT8, DT_FLOAT8_E5M2, DT_FLOAT8_E4M3FN}))
    .OP_END_FACTORY_REG(Slice)

}  // namespace ge

#endif  // NN_COMMON_STUB_OPS_H
