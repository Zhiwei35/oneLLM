#include <cuda_runtime.h>
#include <cuda.h>
#include "src/models/llama/llama_params.h"
#include "src/utils/tensor.h"

void launchAddFusedQKVBiasTransposeAndRoPE(Tensor* q_buf,
                                           Tensor* k_buf,
                                           Tensor* v_buf,
                                           Tensor* QKV,
                                           Tensor* qkv_bias,
                                           Tensor* padding_offset,
                                           Tensor* history_length,
                                           Tensor* input_length,
                                           LLaMAAttentionStaticParams& params);