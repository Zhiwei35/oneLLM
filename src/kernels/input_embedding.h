#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

void launchInputEmbedding(Tensor* input_ids,
                          Tensor* output,
                          Tensor* embed_table);