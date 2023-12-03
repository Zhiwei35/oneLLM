#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include "src/utils/tensor.h"

template<typename T>
void launchInputEmbedding(TensorWrapper<int>* input_ids,    
                          TensorWrapper<T>* output,       
                          EmbeddingWeight<T>* embed_table);