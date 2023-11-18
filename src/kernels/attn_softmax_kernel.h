#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

void launchScaleMaskAndSoftmax(Tensor* qk,
                               Tensor* mask,
                               Tensor* attn_score,
                               float scale);
