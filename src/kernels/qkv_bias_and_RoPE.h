#include <cuda_runtime.h>
#include <cuda.h>
#include "src/models/llama/llama_params.h"
#include "src/utils/tensor.h"
#include "src/weights/base_weights.h"
void launchAddFusedQKVBiasTransposeAndRoPE(Tensor* q_buf,
                                           Tensor* k_buf,
                                           Tensor* v_buf,
                                           Tensor* QKV,
                                           BaseWeight& qkv,
                                           Tensor* padding_offset,
                                           Tensor* history_length,
                                           Tensor* input_length,
                                           LLaMAAttentionStaticParams& params);
