#include "src/kernels/transpose_kernel.h"
#include <iostream>
//[bs, kv head num, max_seq_len, head size]=>[bs, q head num, max_k_len, head size]
//context_length.shape=[bs]
template<typename T>
__global__ void transpose_value_cache(T*          v_dst, 
                                      const T*    v_src,
                                      const size_t layer_offset,
                                      const int    head_num,
                                      const int    q_head_per_kv,
                                      const int    head_size,
                                      const int*   context_length,
                                      const int    max_k_len,
                                      const int    max_seq_len)
{
    const int batch_id = blockIdx.y;
    const int head_id  = blockIdx.z;

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;

    const auto val_src = v_src + layer_offset;
    const auto val_dst = v_dst;

    const auto seq_len = context_length[batch_id];

    const int v_head_size_id = idx % head_size;
    const int v_seq_len_id   = idx / head_size;
    // only fetch context_length(<max_seq_len) kv data from all kv cache of current seq
    if (v_seq_len_id < seq_len) {
        const int64_t src_idx = batch_id * (head_num / q_head_per_kv) * head_size * max_seq_len + //B
                                head_id / q_head_per_kv * head_size * max_seq_len +  // H
                                v_seq_len_id * head_size +                        // s
                                v_head_size_id;                                             // D/x

        const int64_t dst_idx = batch_id * head_num * head_size * max_k_len +  // B
                                head_id * head_size * max_k_len +              // H
                                v_seq_len_id * head_size +                      // s
                                v_head_size_id;                                           // D/x

        val_dst[dst_idx] = val_src[src_idx];
    }
}

template<typename T>
void launchTransposeKVCache(Tensor* k_cache_src,
                            Tensor* v_cache_src,
                            Tensor* context_length,
                            size_t  layer_offset,
                            Tensor* k_cache_dst,
                            Tensor* v_cache_dst
                            )
{
    int batch_size = context_length->shape[0];
    int kv_head_num = k_cache_src->shape[1];
    int max_seq_len = k_cache_src->shape[2];
    int head_num = k_cache_dst->shape[1];
    int max_k_len = k_cache_dst->shape[2];
    int head_size = k_cache_dst->shape[3];

    int q_head_per_kv = head_num / kv_head_num;
    int blockSize = 128;
    dim3 block(128);
    dim3 grid((max_k_len * head_size + blockSize - 1) / blockSize, batch_size, head_num); // q head num
    transpose_value_cache<T><<<grid, block>>>((T*)v_cache_dst, 
                                              (const T*)v_cache_src,
                                              layer_offset,
                                              head_num,
                                              q_head_per_kv,
                                              head_size,
                                              (int*)context_length->data,
                                              max_k_len,
                                              max_seq_len);
                                              
    transpose_value_cache<T><<<grid, block>>>((T*)k_cache_dst, 
                                              (const T*)k_cache_src,
                                              layer_offset,
                                              head_num,
                                              q_head_per_kv,
                                              head_size,
                                              (int*)context_length->data,
                                              max_k_len,
                                              max_seq_len);
}
