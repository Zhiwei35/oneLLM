#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"

void launchTransposeOutRemovePadding(Tensor* qkv_buf_w_pad, 
                                    Tensor* padding_offset,
                                    Tensor* qkv_buf_wo_pad_1);