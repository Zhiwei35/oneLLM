#include <float.h> //FLT_MIN
#include <cuda.h>
#include <stdio.h>
//#include <utils/gpu_config.h>
#include "src/kernels/input_embedding.h"

template<typename T>
__global__ void embeddingFunctor(const int* input_ids,
               T* output, 
               const T* embed_table,
               const int batch_size,
               const int sequeue_length,
               const int hidden_size,
               const int vocab_size)
{
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * sequeue_length * hidden_size;
         index += blockDim.x * gridDim.x) {
        int input_id = input_ids[index / hidden_size];
        output[index] = embed_table[input_id * hidden_size + index % hidden_size];
        if (index == 0){
            printf("embedding res: \n");
            printf("%f\n",output[index]);
        }
    }
}

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // INT [batch_size, sequenue_length]
                          TensorWrapper<T>* output,       // FP32 [batch_size, sequeue_length, hidden_size]
                          EmbeddingWeight<T>* embed_table,// FP32 [vocal_size, hidden_size]
                          int vocab_size) {//consider add shape attr in embeddingweight to avoid vocab size input 
    const int blockSize = 256;
    const int batch_size = output->shape[0];
    const int sequeue_length = output->shape[1];
    const int hidden_size = output->shape[2];
    const int gridSize = (blockSize + output->size() - 1) / blockSize;
    // printf("calling input embedding\n");
    embeddingFunctor<T><<<gridSize, blockSize>>>(input_ids->data,
                                                 output->data,
                                                 embed_table->data,
                                                 batch_size,
                                                 sequeue_length,
                                                 hidden_size,
                                                 vocab_size);
    // printf("called input embedding\n");
}

template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<float>* output,       
                                   EmbeddingWeight<float>* embed_table,
                                   int vocab_size);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<half>* output,       
                                   EmbeddingWeight<half>* embed_table,
                                   int vocab_size);