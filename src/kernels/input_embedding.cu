#include <float.h> //FLT_MIN
#include <cuda.h>
#include <iostream>
//#include <utils/gpu_config.h>
#include "src/kernels/input_embedding.h"

__global__ void embeddingFunctor(const int* input_ids,
               float* output, 
               const float* embed_table,
               const int batch_size,
               const int sequeue_length,
               const int hidden_size,
               const int vocab_size)
{
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * sequeue_length * hidden_size;
         index += blockDim.x * gridDim.x) {
        int input_id = input_ids[index / hidden_size];
        output[index] = embed_table[input_id * hidden_size + index % hidden_size];
    }
}


void launchInputEmbedding(Tensor* input_ids,    // INT [batch_size, sequenue_length]
                          Tensor* output,       // FP32 [batch_size, sequeue_length, hidden_size]
                          Tensor* embed_table)  // FP32 [vocal_size, hidden_size]
{
    const int blockSize = 256;
    const int batch_size = output->shape[0];
    const int sequeue_length = output->shape[1];
    const int hidden_size = output->shape[2];
    const int vocab_size = embed_table->shape[0];
    const int gridSize = (blockSize + output->size() - 1) / blockSize;
    embeddingFunctor<<<gridSize, blockSize>>>((int*) input_ids->data,
                                              (float*) output->data,
                                              (float*) embed_table->data,
                                              batch_size,
                                              sequeue_length,
                                              hidden_size,
                                              vocab_size);
}