#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

template<typename T>
void launchScaleMaskAndSoftmax(Tensor* qk,
                               Tensor* mask,
                               Tensor* attn_score,
                               float scale);

void launchScaleMaskAndSoftmax<float>(Tensor* qk,
                                    Tensor* mask,
                                    Tensor* attn_score,
                                    float scale);