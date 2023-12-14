#include <iostream>
#include "src/kernels/fused_transpose_reshape_remv_pad.h"
// [b,h,s,d]=>[b,s,h,d]=>[num tokens,h,d]
// padding_offset.shape = [num_tokens]
template<typename T>
__global__ void fused_transpose_reshape_remv_pad(T*           src,
                                                T*           dst,
                                                const int    num_tokens,
                                                const int    batch_size,
                                                const int    seq_len,
                                                const int    head_num,
                                                const int    head_size,
                                                const int*   padding_offset/*for remove padding*/)
{
    int token_id = blockIdx.x; // token nums
    // map to input id
    int batch_id = (blockIdx.x + padding_offset[token_id]) / seq_len;
    int seq_id = (blockIdx.x + padding_offset[token_id]) % seq_len;
    // transpose
    int src_offset = batch_id * head_num * seq_len * head_size + seq_id * head_size;
    int dst_offset = token_id * head_num * head_size;

    for (int i = threadIdx.x; i < head_num *  head_size; i += blockDim.x) {
        int head_id = i / head_size;
        int head_size_id = i % head_size;
        dst[dst_offset + i] = src[src_offset + head_id * seq_len * head_size + head_size_id];
        if (i == 0) {
            printf("context attention top1 res: \n");
            printf("%f\n",dst[dst_offset + i]);
        }
    }
}
template<typename T>
void launchTransposeOutRemovePadding(TensorWrapper<T>* qkv_buf_w_pad, 
                                     TensorWrapper<int>* padding_offset,
                                     TensorWrapper<T>* qkv_buf_wo_pad_1)
{
    int batch_size = qkv_buf_w_pad->shape[0];
    int head_num = qkv_buf_w_pad->shape[1];
    int seq_len = qkv_buf_w_pad->shape[2];
    int head_size = qkv_buf_w_pad->shape[3];
    int num_tokens = qkv_buf_wo_pad_1->shape[0];
    dim3 grid(num_tokens);
    dim3 block(std::min(head_num * head_size, 1024));
    // std::cout << "calling remove pad kernel" << "\n";
    fused_transpose_reshape_remv_pad<T><<<grid, block>>>(qkv_buf_w_pad->data,
                                                      qkv_buf_wo_pad_1->data,
                                                      num_tokens,
                                                      batch_size,
                                                      seq_len,
                                                      head_num,
                                                      head_size,
                                                      padding_offset->data);
    // std::cout << "called remove pad kernel" << "\n";

}

template void launchTransposeOutRemovePadding(TensorWrapper<float>* qkv_buf_w_pad, 
                                              TensorWrapper<int>* padding_offset,
                                              TensorWrapper<float>* qkv_buf_wo_pad_1);
template void launchTransposeOutRemovePadding(TensorWrapper<half>* qkv_buf_w_pad, 
                                              TensorWrapper<int>* padding_offset,
                                              TensorWrapper<half>* qkv_buf_wo_pad_1);

