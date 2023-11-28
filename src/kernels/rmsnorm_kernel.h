#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"
#include "src/weights/llama/norm_weights.h"

void launchRMSNorm( Tensor* decoder_out, // [num tokens, hidden_units]
                    LayerNormWeight& attn_norm_weight, //RMSNorm weights
                    float eps //RMSNorm eps
                    );
