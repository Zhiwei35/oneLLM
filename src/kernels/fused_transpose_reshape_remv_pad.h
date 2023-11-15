#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"
template<typename T>
void launchTransposeOutRemovePadding(Tensor<T>* qkv_buf_w_pad, 
                                    Tensor<T>* padding_offset,
                                    Tensor<T>* qkv_buf_wo_pad_1);