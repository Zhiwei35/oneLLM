#include "src/kernels/transpose_kernel.h"
#include <iostream>
//if MQA or GQA, we should use this transpose to broadcast kv head num to q head num
//[num layers, bs, kv head num, max_seq_len, head size]=>[bs, q head num, max_k_len, head size]
//context_length.shape=[bs]
// 这个kernel叫repeat_interleave或者broadcast比较合理
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
void launchTransposeKVCache(TensorWrapper<T>* k_cache_src,
                            TensorWrapper<T>* v_cache_src,
                            TensorWrapper<int>* context_length,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<T>* k_cache_dst,
                            TensorWrapper<T>* v_cache_dst
                            )
{
    int batch_size = context_length->shape[0];
    int kv_head_num = k_cache_src->shape[1];
    int max_seq_len = k_cache_src->shape[2];
    int head_num = k_cache_dst->shape[1];
    
    int max_k_len = k_cache_dst->shape[2];
    int head_size = k_cache_dst->shape[3];
    int layer = layer_id->getVal();
    //note: here MUSTN'T use layer_id->getVal<int>(), because we cant access GPU memory directly by [] if data is on GPU
    //note: so we can make layer data locate on CPU
    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    int q_head_per_kv = head_num / kv_head_num;
    int blockSize = 128;
    dim3 block(128);
    dim3 grid((max_k_len * head_size + blockSize - 1) / blockSize, batch_size, head_num); // q head num
    std::cout << "calling transpose/broadcast kernel" << "\n";    
    transpose_value_cache<T><<<grid, block>>>(v_cache_dst->data, 
                                              v_cache_src->data,
                                              layer_offset,
                                              head_num,
                                              q_head_per_kv,
                                              head_size,
                                              context_length->data,
                                              max_k_len,
                                              max_seq_len);
                                              
    transpose_value_cache<T><<<grid, block>>>(k_cache_dst->data, 
                                              k_cache_src->data,
                                              layer_offset,
                                              head_num,
                                              q_head_per_kv,
                                              head_size,
                                              context_length->data,
                                              max_k_len,
                                              max_seq_len);
    std::cout << "called transpose/broadcast kernel" << "\n";

}

template void launchTransposeKVCache(TensorWrapper<float>* k_cache_src,
                            TensorWrapper<float>* v_cache_src,
                            TensorWrapper<int>* context_length,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<float>* k_cache_dst,
                            TensorWrapper<float>* v_cache_dst
                            );
template void launchTransposeKVCache(TensorWrapper<half>* k_cache_src,
                            TensorWrapper<half>* v_cache_src,
                            TensorWrapper<int>* context_length,
                            TensorWrapper<int>* layer_id,
                            TensorWrapper<half>* k_cache_dst,
                            TensorWrapper<half>* v_cache_dst
                            );