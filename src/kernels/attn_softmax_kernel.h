#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template<typename T>
void launchScaleMaskAndSoftmax(TensorWrapper<T>* qk,
                               TensorWrapper<uint8_t>* mask,
                               TensorWrapper<T>* attn_score,
                               float scale);
