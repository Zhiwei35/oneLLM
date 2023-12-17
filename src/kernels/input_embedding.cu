#include <float.h> //FLT_MIN
#include <cuda.h>
#include <stdio.h>
//#include <utils/gpu_config.h>
#include "src/kernels/input_embedding.h"

template<typename T>
__global__ void embeddingFunctor(const int* input_ids,
               T* output, 
               const T* embed_table,
               const int max_context_token_num,
               //const int sequeue_length,
               const int hidden_size,
               const int vocab_size)
{
    // for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < batch_size * sequeue_length * hidden_size;
    //      index += blockDim.x * gridDim.x) {
    //     int input_id = input_ids[index / hidden_size];
    //     output[index] = embed_table[input_id * hidden_size + index % hidden_size];
    //     if (index == 0){
    //         printf("embedding res: \n");
    //         printf("%f\n",output[index]);
    //     }
    // }
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    while (index < max_context_token_num * hidden_size) {
        int id = input_ids[index / hidden_size];
        output[index] = embed_table[id * hidden_size + index % hidden_size];
        index += blockDim.x * gridDim.x;
    }
    if (index == 0){
        printf("embedding res: \n");
        printf("%f\n",output[index]);
    }
}

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    // INT [max context token num]
                          TensorWrapper<T>* output,       // FP32 [max context token num, hidden_size] = [max seq len, 4096]
                          EmbeddingWeight<T>* embed_table,// FP32 [vocal_size, hidden_size]
                          int vocab_size) {//consider add shape attr in embeddingweight to avoid vocab size input 
    const int blockSize = 256;
    // const int batch_size = 1;
    const int max_context_token_num = output->shape[0];
    const int hidden_size = output->shape[1];
//    const int gridSize = (blockSize + output->size() - 1) / blockSize;
    const int gridSize = 2048;
    ONELLM_CHECK_WITH_INFO(max_context_token_num == input_ids->shape[0], "input ids 1st shape should equal to 1st shape of output");
    printf("calling input embedding\n");
    printf("context decoder input shape:\n");
    printf("%d, %d\n", max_context_token_num, hidden_size);
    printf("block num = %d, thread num = %d\n", gridSize, blockSize);
    embeddingFunctor<T><<<gridSize, blockSize>>>(input_ids->data,
                                                 output->data,
                                                 embed_table->data,
                                                 max_context_token_num,
                                                 //sequeue_length,
                                                 hidden_size,
                                                 vocab_size);
    printf("called input embedding\n");
}

template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<float>* output,       
                                   EmbeddingWeight<float>* embed_table,
                                   int vocab_size);
template void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                                   TensorWrapper<half>* output,       
                                   EmbeddingWeight<half>* embed_table,
                                   int vocab_size);
