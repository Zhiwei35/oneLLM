#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"
#include "src/models/llama/llama_params.h"
#include "src/weights/base_weights.h"

void launchDecoderMaskedMHA(Tensor* qkv_buf,
                            BaseWeight& qkv,
                            Tensor* k_cache,
                            Tensor* v_cache,
                            Tensor* finished,
                            Tensor* step,
                            Tensor* mha_output,
                            LLaMAAttentionStaticParams& static_params);
