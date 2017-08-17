/*!
 * Copyright (c) 2015 by Contributors
 * \file softmax_output.cu
 * \brief
 * \author Bing Xu
*/
#include "./softmax_focal_output-inl.h"
#include "../mxnet_op.h"
#include "../../common/cuda_utils.h"

#define SOFTMAXFOCAL_CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

namespace mshadow {
namespace cuda {

    template<typename DType>
    __global__ void SoftmaxFocalGradKernel(DType *dst,
                                           const DType  *src,
                                           const DType  *label,
                                           const int gamma,
                                           const DType *alphas,
                                           const int line_size,
                                           const int num
                                             ) {
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (y >= num) {
            return;
        }
        const index_t k = static_cast<int>(label[y]);
        const DType scale = pow((1 - src[y*line_size+ k]), gamma) * alphas[k];
        for (index_t x = 0; x < line_size; ++x) {
          if (x == k) {
            dst[y*line_size+k] = scale*(src[y*line_size+ k] - 1.0f);
          } else {
            dst[y*line_size+ x] = scale*src[y*line_size+ x];
          }
        }
    }


   template<typename DType>
    __global__ void SoftmaxFocalGradKernel(DType *dst,
                                           const DType  *src,
                                           const DType  *label,
                                           const DType ignore_label,
                                           const int gamma,
                                           const DType *alphas,
                                           const int line_size,
                                           const int num
                                             ) {
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        
        if (y >= num) {
            return;
        }
        const index_t k = static_cast<int>(label[y]);
        const DType scale = pow((1 - src[y*line_size+ k]), gamma) * alphas[k];
        for (index_t x = 0; x < line_size; ++x) {
          if (static_cast<int>(ignore_label) == k) {
            dst[y*line_size+k] = 0.0f;
          }else{
            if (x == k) {
                dst[y*line_size+k] = scale*(src[y*line_size+ k] - 1.0f);
            } else {
                dst[y*line_size+ x] = scale*src[y*line_size+ x];
            }
          }
        }
    }


   template<typename DType>
    __global__ void SoftmaxFocalMultiGradKernel(DType *dst,
                                           const DType  *src,
                                           const DType  *label,
                                           const int gamma,
                                           const DType *alphas,
                                           const int class_num,
                                           const int sample_out_size,
                                           const int num
                                             ) {
      
       int page_size = sample_out_size * class_num;
        CUDA_KERNEL_LOOP(i,num){
            int n = i%sample_out_size;
            int y = i/sample_out_size;
            const index_t k = static_cast<int>(label[y*sample_out_size + n]);
            DType scale = pow((1 - src[y*page_size + k*sample_out_size + n]), gamma) * alphas[k];
            for (index_t x = 0; x < class_num; ++x) {
                if (x == k) {
                  dst[y*page_size + k*sample_out_size + n] = scale*(src[y*page_size + k*sample_out_size + n] - 1.0f);
                } else {
                  dst[y*page_size + x*sample_out_size + n] = scale*src[y*page_size + x*sample_out_size + n];
                } 
            }
        }
    }

    template<typename DType>
        __global__ void SoftmaxFocalMultiGradKernel(DType *dst,
                                               const DType  *src,
                                               const DType  *label,
                                               const DType   ignore_label,
                                               const int gamma,
                                               const DType *alphas,
                                               const int class_num,
                                               const int sample_out_size,
                                               const int num
                                                 ) {
          
           int page_size = sample_out_size * class_num;
            CUDA_KERNEL_LOOP(i,num){
                int n = i%sample_out_size;
                int y = i/sample_out_size;
                const index_t k = static_cast<int>(label[y*sample_out_size + n]);
                DType scale = pow((1 - src[y*page_size + k*sample_out_size + n]), gamma) * alphas[k];

                if (k == static_cast<int>(ignore_label)) {
                    for (index_t x = 0; x < class_num; ++x) {
                           dst[y*page_size + x*sample_out_size + n] = DType(0.0f);
                    }
                } else{
                   for (index_t x = 0; x < class_num; ++x) {
                        if (x == k) {
                          dst[y*page_size + k*sample_out_size + n] = scale*(src[y*page_size + k*sample_out_size + n] - 1.0f);
                        } else {
                          dst[y*page_size + x*sample_out_size + n] = scale*src[y*page_size + x*sample_out_size + n];
                        } 
                   }
               }

            }
    }

  }// cuda

}//mshadow

namespace mshadow{

    template<typename DType>
    inline void SoftmaxFocalGrad(Tensor<gpu, 2, DType> dst,
                            const Tensor<gpu, 2, DType> &src,
                            const Tensor<gpu, 1, DType> &label,
                            const int gamma,
                            Tensor<gpu, 1 ,DType> alphas) {

     CHECK_EQ(dst.CheckContiguous(), true);
     cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
     const int num_thread = cuda::kMaxThreadsPerBlock;
     int num = dst.size(0);
     dim3 dimBlock(num_thread);
     dim3 dimGrid((num - 1) / num_thread + 1);
     DType *out_ptr = dst.dptr_;
     
     cuda::CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocal Forward");
     
     cuda::SoftmaxFocalGradKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(out_ptr,
                                                                           src.dptr_,
                                                                           label.dptr_,
                                                                           gamma,
                                                                           alphas.dptr_,
                                                                           dst.size(1),
                                                                           num);
     SOFTMAXFOCAL_CUDA_CHECK(cudaPeekAtLastError());
    }  

    template<typename DType>
    inline void SoftmaxFocalGrad(Tensor<gpu, 2, DType> dst,
                            const Tensor<gpu, 2, DType> &src,
                            const Tensor<gpu, 1, DType> &label,
                            const DType &ignore_label, 
                            const int gamma,
                            const Tensor<gpu, 1 ,DType> alphas)
    {
         CHECK_EQ(dst.CheckContiguous(), true);
         cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
         const int num_thread = cuda::kMaxThreadsPerBlock;
         int num = dst.size(0);
         dim3 dimBlock(num_thread);
         dim3 dimGrid((num - 1) / num_thread + 1);
         DType *out_ptr = dst.dptr_;
         
         cuda::CheckLaunchParam(dimGrid, dimBlock, "SoftmaxFocal Forward");
         
         cuda::SoftmaxFocalGradKernel<DType><<<dimGrid, dimBlock, 0, stream>>>(out_ptr,
                                                                               src.dptr_,
                                                                               label.dptr_,
                                                                               ignore_label,
                                                                               gamma,
                                                                               alphas.dptr_,
                                                                               dst.size(1),
                                                                               num);
         SOFTMAXFOCAL_CUDA_CHECK(cudaPeekAtLastError());
    }   
      

     
      
    template<typename DType>
    inline void SoftmaxFocalGrad(Tensor<gpu, 3, DType> dst,
                            const Tensor<gpu, 3, DType> &src,
                            const Tensor<gpu, 2, DType> &label,
                            const int gamma,
                            const Tensor<gpu, 1 ,DType> alphas) {

         CHECK_EQ(dst.CheckContiguous(), true);
         cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
         int num_kernels = dst.size(0)*dst.size(2);


         DType *out_ptr = dst.dptr_;
         using namespace mxnet::op::mxnet_op; 
         cuda::SoftmaxFocalMultiGradKernel<DType><<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
              0, stream>>>(out_ptr,
                           src.dptr_,
                           label.dptr_,
                           gamma,
                           alphas.dptr_,
                           dst.size(1),
                           dst.size(2),
                           num_kernels);
         SOFTMAXFOCAL_CUDA_CHECK(cudaPeekAtLastError());

    }  



    template<typename DType>
    inline void SoftmaxFocalGrad(Tensor<gpu, 3, DType> dst,
                            const Tensor<gpu, 3, DType> &src,
                            const Tensor<gpu, 2, DType> &label,
                            const DType &ignore_label,
                            const int gamma,
                            const Tensor<gpu, 1 ,DType> alphas) {
        CHECK_EQ(dst.CheckContiguous(), true);
        cudaStream_t stream = Stream<gpu>::GetStream(dst.stream_);
        int num_kernels = dst.size(0)*dst.size(2);

        DType *out_ptr = dst.dptr_;
        using namespace mxnet::op::mxnet_op; 
        cuda::SoftmaxFocalMultiGradKernel<DType><<<cuda_get_num_blocks(num_kernels), mshadow::cuda::kBaseThreadNum,
             0, stream>>>(out_ptr,
                          src.dptr_,
                          label.dptr_,
                          ignore_label,
                          gamma,
                          alphas.dptr_,
                          dst.size(1),
                          dst.size(2),
                          num_kernels);
        SOFTMAXFOCAL_CUDA_CHECK(cudaPeekAtLastError());

    }
} // namespace mshadow

namespace mxnet {
namespace op {

template<>
Operator *CreateOp<gpu>(SoftmaxFocalOutputParam param, int dtype) {
  Operator *op = NULL;
  MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
    op = new SoftmaxFocalOutputOp<gpu, DType>(param);
  })
  return op;
}

}  // namespace op
}  // namespace mxnet

