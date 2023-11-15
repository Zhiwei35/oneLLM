#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

template<typename T,typename T1>
void launchScaleMaskAndSoftmax(Tensor<T>* qk,
                               Tensor<T1>* mask,
                               Tensor<T>* attn_score,
                               float scale);
