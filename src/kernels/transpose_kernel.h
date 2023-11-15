#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

template<typename T>
void launchTransposeKVCache(Tensor<T>* k_cache_src,
                            Tensor<T>* v_cache_src,
                            Tensor<T>* context_length,
                            size_t  layer_offset,
                            int q_head_per_kv,
                            Tensor<T>* k_cache_dst,
                            Tensor<T>* v_cache_dst
                            );
