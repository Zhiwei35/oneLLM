#include <cuda_runtime.h>
#include <cuda.h>
void launchDecoderMaskedMHA(Tensor* qkv_buf,
                            Tensor* k_cache,
                            Tensor* v_cache,
                            Tensor* finished,
                            Tensor* step,
                            Tensor* mha_output);
