#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include "src/kernels/cublas_wrapper.h"
#include "src/utils/tensor.h"
#include "src/weights/llama/attention_weights.h"
#include "src/utils/macro.h"
//TODO: when enable int8/int4 weight only, we can add a new type param T2 to represent weight type
void launchLinearGemm(Tensor* input,
                      BaseWeight& weight, 
                      Tensor* output,
                      bool trans_a = false,
                      bool trans_b = false,
                      bool shared_out_buf = false);

void launchLinearStridedBatchGemm(Tensor* input1,
                                  Tensor* input2,
                                  Tensor* output,
                                  bool trans_a = false,
                                  bool trans_b = false);
