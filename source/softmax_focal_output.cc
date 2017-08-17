/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cc
 * \brief
 * \author Bing Xu
*/
#include "./softmax_focal_output-inl.h"
#include "mshadow/tensor.h"
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <cstring>
#include <map>
#include <string>
#include <vector>
#include <utility>
#include <math.h>

namespace mshadow {
    
    template<typename DType>
    inline void SoftmaxFocalGrad(Tensor<cpu, 2, DType> dst,
                            const Tensor<cpu, 2, DType> &src,
                            const Tensor<cpu, 1, DType> &label,
                            const int gamma,
                            const Tensor<cpu, 1 ,DType> alphas) {
    #pragma omp parallel for
      for (openmp_index_t y = 0; y < dst.size(0); ++y) {
        const index_t k = static_cast<int>(label[y]);
        const DType scale = pow((1 - src[y][k]), gamma) * alphas[k];
        for (index_t x = 0; x < dst.size(1); ++x) {
          if (x == k) {
            dst[y][k] = scale*(src[y][k] - 1.0f);
          } else {
            dst[y][x] = scale*src[y][x];
          }
        }
      }
    }


    template<typename DType>
    inline void SoftmaxFocalGrad(Tensor<cpu, 2, DType> dst,
                            const Tensor<cpu, 2, DType> &src,
                            const Tensor<cpu, 1, DType> &label,
                            const DType &ignore_label, 
                            const int gamma,
                            const Tensor<cpu, 1 ,DType> alphas){
    #pragma omp parallel for
      for (openmp_index_t y = 0; y < dst.size(0); ++y) {
        const index_t k = static_cast<int>(label[y]);
        const DType scale = pow((1 - src[y][k]), gamma) * alphas[k];
        for (index_t x = 0; x < dst.size(1); ++x) {
          if (static_cast<int>(ignore_label) == k) {
            dst[y][x] = 0.0f;
          } else {
            if (x == k) {
              dst[y][k] = scale*(src[y][k] - 1.0f);
            } else {
              dst[y][x] = scale * src[y][x];
            }
          }
        }
      }
    }


    template<typename DType>
    inline void SoftmaxFocalGrad(Tensor<cpu, 3, DType> dst,
                            const Tensor<cpu, 3, DType> &src,
                            const Tensor<cpu, 2, DType> &label,
                            const int gamma,
                           const Tensor<cpu, 1 ,DType> alphas) {
    #pragma omp parallel for
      for (openmp_index_t n = 0; n < dst.size(2); ++n) {
        for (index_t y = 0; y < dst.size(0); ++y) {
          const index_t k = static_cast<int>(label[y][n]);
          DType scale = pow((1 - src[y][k][n]), gamma) * alphas[k];
          for (index_t x = 0; x < dst.size(1); ++x) {
            if (x == k) {
              dst[y][k][n] = scale*(src[y][k][n] - 1.0f);
            } else {
              dst[y][x][n] = scale*src[y][x][n];
            }
          }
        }
      }
    }


    template<typename DType>
    inline void SoftmaxFocalGrad(Tensor<cpu, 3, DType> dst,
                            const Tensor<cpu, 3, DType> &src,
                            const Tensor<cpu, 2, DType> &label,
                            const DType &ignore_label,
                            const int gamma,
                            const Tensor<cpu, 1 ,DType> alphas) {
    #pragma omp parallel for
      for (openmp_index_t n = 0; n < dst.size(2); ++n) {
        for (index_t y = 0; y < dst.size(0); ++y) {
          const index_t k = static_cast<int>(label[y][n]);
          DType scale = pow((1 - src[y][k][n]), gamma) * alphas[k];
          if (k == static_cast<int>(ignore_label)) {
            for (index_t x = 0; x < dst.size(1); ++x) {
              dst[y][x][n] = DType(0.0f);
            }
          } else {
            for (index_t x = 0; x < dst.size(1); ++x) {
              if (x == k) {
                dst[y][k][n] =scale* (src[y][k][n] - 1.0f);
              } else {
                dst[y][x][n] =scale * src[y][x][n];
              }
            }
          }
        }
      }
    }

}


namespace mxnet {
namespace op {


template<>
Operator *CreateOp<cpu>(SoftmaxFocalOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxFocalOutputOp<cpu, DType>(param);
  })
  return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *SoftmaxFocalOutputProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                     std::vector<int> *in_type) const {
  std::vector<TShape> out_shape, aux_shape;
  std::vector<int> out_type, aux_type;
  CHECK(InferType(in_type, &out_type, &aux_type));
  CHECK(InferShape(in_shape, &out_shape, &aux_shape));
  DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(SoftmaxFocalOutputParam);

MXNET_REGISTER_OP_PROPERTY(SoftmaxFocalOutput, SoftmaxFocalOutputProp)
.describe(R"code(Softmax with logit loss.

In the forward pass, the softmax output is returned. Assume the input data has
shape *(n,k)*, then the output will have the same shape as the input, which is computed by

.. math::
   out[i,:] = softmax(data[i,:])

for :math:`i=0,...,n-1`, where

.. math::
   softmax(x) = \left[..., \frac{exp(x[j])}{exp(x[0])+...+exp(x[k-1])}, ...\right]

For general *N*-D input array with shape :math:`(d_1, ..., d_n)`. Denoted by the size
:math:`s=d_1d_2...d_n`. The way to compute softmax various:

- ``preserve_shape`` is false (default). Reshape input into a 2-D array with
  shape :math:`(d_1, s/d_1)` beforing computing the softmax, and then reshaped back to the
  original shape.

- ``preserve_shape`` is true. For all :math:`i_1, ..., i_{n-1}`, compute

  .. math::
    out[i_1, ..., i_{n-1}, :] = softmax(data[i_1, ..., i_{n-1},:])

- ``multi_output`` is true. For all :math:`i_1, ..., i_{n-1}`, compute

  .. math::
    out[i_1, :, ..., i_{n-1}] = softmax(data[i_1, :, ..., i_{n-1}])

In the backward pass, the logit loss, also called cross-entroy loss, is
added. The provided label can be a *(N-1)*-D label index array or a *N*-D label
probability array.

Examples with a particular label can be ignored during backward by specifying
``ignore_label`` (also need ``use_ignore`` to be true).

A scale can be applied to the gradient by ``grad_scale``, which is often used in
mutli-loss object function in which we can given each loss different weight. It
also supports various ways to normalize the gradient by ``normalization``:

- **null**: do nothing
- **batch**: divide by batch size (number of examples)
- **valid**: divide by the number of examples which are not ignored.
)code" ADD_FILELINE)
.add_argument("data", "ndarray-or-symbol", "Input data.")
.add_argument("label", "ndarray-or-symbol", "Ground truth label.")
.add_arguments(SoftmaxFocalOutputParam::__FIELDS__());

}  // namespace op
}  // namespace mxnet
