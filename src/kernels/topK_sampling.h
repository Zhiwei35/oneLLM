#include <curand.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "src/utils/tensor.h"
#include "src/utils/params.h"

void launchSampling(Tensor* topk_id,
                    Tensor* topk_val,
                    Tensor* seqlen,
                    Tensor* is_finished,
                    Tensor* output_id,
                    IntDict& params);