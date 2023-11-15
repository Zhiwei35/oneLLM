#include <cuda_runtime.h>
#include <cuda.h>
#include "src/kernels/cublas_wrapper.h"
#include "src/utils/tensor.h"
#include "src/weights/llama/attention_weights.h"

//TODO: when enable int8/int4 weight only, we can add a new type param T2 to represent weight type
template<typename T>
void launchLinearGemm(Tensor<T>* input,
                      BaseWeight<T>& weight, 
                      Tensor<T>* output);

template<typename T>
void launchLinearStridedBatchGemm(Tensor<T>* input1,
                                  Tensor<T>* input2,
                                  Tensor<T>* output);
