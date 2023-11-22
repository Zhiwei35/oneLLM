#include <float.h> //FLT_MIN
#include <cuda.h>
#include <iostream>
//#include <utils/gpu_config.h>
#include "src/kernels/input_embedding.h"
#include "src/utils/tensor.h"

__global__ void embeddingFunctor(const int* input_ids,
               float* output, 
               const float* embed_table, // this->weight["model.embed_tokens.weight"]
               const int sequeue_length,
               const int hidden_size,
               const int vocab_size) {
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < sequeue_length * hidden_size;
         index += blockDim.x * gridDim.x) {
        int input_id = input_ids[index / hidden_size];
        output[index] = embed_table[input_id * hidden_size + index % hidden_size];
    }
}


void launchInputEmbedding(const int* input_ids, float* output, float* embed_table, const int sequeue_length, const int hidden_size, const int vocab_size) {
    const int blockSize = 256;
    const int gridSize = (blockSize + sequeue_length * hidden_size - 1) / blockSize;
    embeddingFunctor<<<gridSize, blockSize>>>(input_ids, output, embed_table, sequeue_length, hidden_size, vocab_size);
}