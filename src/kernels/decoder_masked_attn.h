#include <cuda_runtime.h>
#include <cuda.h>
void launchDecoderMaskedMHA(float* q,
                            float* k,
                            float* v,
                            float* k_cache,
                            float* v_cache,
                            float* mha_output,
                            const int batch_size,
                            const int num_heads,
                            const int head_size,
                            const int step);
