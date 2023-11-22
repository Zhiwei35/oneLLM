#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

void launchTransposeKVCache(Tensor* k_cache_src,
                            Tensor* v_cache_src,
                            Tensor* context_length,
                            Tensor*  layer_id,
                            Tensor* k_cache_dst,
                            Tensor* v_cache_dst
                            );
