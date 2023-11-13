#include <float.h> //FLT_MIN
#include <cuda.h>
#include <iostream>
//#include <utils/gpu_config.h>
#include "src/kernels/beamsearch_topK.h"
#include <cub/cub.cuh>

template<typename T>
__global__ void embedding_functor(T* output, 
               const T* embed_table, // this->weight["model.embed_tokens.weight"]
               const int* input_ids,
               const int sequeue_length,
               const int hidden_size,
               const int vocab_size) {
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < sequeue_length * hidden_size;
         index += blockDim.x * gridDim.x) {
        input_id = input_ids[index / hidden_size];
        output[index] = embed_table[input_id * hidden_size + index % hidden_size];
    }
}

