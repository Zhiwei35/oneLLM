#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

void launchInputEmbedding(const int* input_ids, float* output, float* embed_table, const int sequeue_length, const int hidden_size, const int vocab_size);
