#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/kernels/cublas_wrapper.h"
#include "src/utils/tensor.h"
#include "src/weights/llama/attention_weights.h"
#include "src/utils/macro.h"
//TODO: when enable int8/int4 weight only, we can add a new type param T2 to represent weight type
template<typename T>
void launchLinearGemm(TensorWrapper<T>* input,
                      BaseWeight<T>& weight, 
                      TensorWrapper<T>* output,
                      cublasWrapper* cublas_wrapper,
                      bool trans_a = false,
                      bool trans_b = false,
                      bool shared_out_buf = false,
                      int cur_input_len = 1);
template<typename T>
void launchLinearStridedBatchGemm(TensorWrapper<T>* input1,
                                  TensorWrapper<T>* input2,
                                  TensorWrapper<T>* output,
                                  cublasWrapper* cublas_wrapper,
                                  bool trans_a = false,
                                  bool trans_b = false);
