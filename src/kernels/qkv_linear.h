#include <cuda_runtime.h>
#include <cuda.h>
#include "src/kernels/cublas_wrapper.h"
#include "src/utils/tensor.h"

//TODO: when enable int8/int4 weight only, we can add a new type param T2 to represent weight type
template<typename T>
void launchLinearGemm(Tensor* input,
                      BaseWeight<T>& weight, 
                      Tensor* output);

template<typename T>
void launchLinearStridedBatchGemm(Tensor* input1,
                                  Tensor* input2,
                                  Tensor* output);
