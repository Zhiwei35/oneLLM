#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/utils/vectorize_utils.h"

template<typename T>
void launchAct(const TensorWrapper<T>* input, TensorWrapper<T>* out);