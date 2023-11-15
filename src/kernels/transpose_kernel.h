#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

template<typename T>
void launchTransposeKVCache(Tensor* k_cache_src,
                            Tensor* v_cache_src,
                            Tensor* context_length,
                            size_t  layer_offset,
                            int q_head_per_kv,
                            Tensor* k_cache_dst,
                            Tensor* v_cache_dst
                            );
