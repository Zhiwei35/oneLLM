#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"
#include "src/weights/llama/norm_weights.h"

template<typename T>
void launchRMSNorm( TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                    LayerNormWeight<T>& attn_norm_weight, //RMSNorm weights
                    float eps //RMSNorm eps
                    );
