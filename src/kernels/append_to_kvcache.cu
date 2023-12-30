// k/v shape = [bs, kv_head num, max_q_len, head size] // 为什么这里不是max_k_len?因为k v=w * x，此时x中seqlen维度为max_q_len
// kv cache shape = [num layers, bs, kv_head num, max_seq_len, head size] = >[bs, kv_head num, seqlen[history_len:history_len+seqlen] , head size]
// kv cache 是每个layer都有单独的kv cache ， from llama_from_ft.cc#104
// ksrc shape = [bs, kv_head num, max_q_len, head size],为什么是q_len?

#include "src/kernels/append_to_kvcache.h"
#include<iostream>
template<typename T>
__global__ void append_key_cache(T*          k_dst, //[num layers, bs, kv head num, max_q_len, head size]
                                 const size_t layer_offset,
                                 const T*     k_src,
                                 const int    kv_head_num,
                                 const int    head_size,
                                 const int*   cur_query_length,
                                 const int*   history_length,
                                 const int    max_q_len, 
                                 const int    max_seq_len){
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tid = threadIdx.x;
    int token_id = blockIdx.x;
    
    // 当前layer的k cache
    T* k_cache_dst = k_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id];
    //note: the if judge is a must, because the max_q_len is GTE than cur_seq_len.
    if(token_id < cur_seq_len){
    // [batch, head num, max_q_len, head size] -> [batch, head num, maxseqlen[cumsum_seq_len:cumsum_seq_len+cur_seq_len], head size]
        int src_offset = batch_id * kv_head_num * max_q_len * head_size + //为什么这里不是max_k_len，新进来的kv应该是max_k_len和max_v_len
                            head_id * max_q_len * head_size + 
                                token_id * head_size + tid;
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size +
                            head_id * max_seq_len * head_size + 
                                (cumsum_seq_len + token_id) * head_size + tid;
        k_cache_dst[dst_offset] = k_src[src_offset];
    }
}

template<typename T>
__global__ void append_value_cache(T*          v_dst,
                                    const size_t layer_offset,
                                    const T*     v_src,
                                    const int    kv_head_num,
                                    const int    head_size,
                                    const int*   cur_query_length,
                                    const int*   history_length,
                                    const int    max_q_len, 
                                    const int    max_seq_len)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tid = threadIdx.x;
    int token_id = blockIdx.x;
    
    // 当前layer的v cache
    T* v_cache_dst = v_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id];
    //note: the if judge is a must, because the max_q_len is GTE than cur_seq_len.
    if(token_id < cur_seq_len){
    // [batch, head num, max_q_len, head size] -> [batch, head num, maxseqlen[cumsum_seq_len:cumsum_seq_len+cur_seq_len], head size]
        int src_offset = batch_id * kv_head_num * max_q_len * head_size + 
                            head_id * max_q_len * head_size + 
                                token_id * head_size + tid;
        int dst_offset = batch_id * kv_head_num * max_seq_len * head_size +
                            head_id * max_seq_len * head_size + 
                                (cumsum_seq_len + token_id) * head_size + tid;
        v_cache_dst[dst_offset] = v_src[src_offset];
    }
}
// k/v shape = [bs, kv_head num, max_q_len, head size] // 为什么这里不是max_k_len，新进来的kv应该是max_k_len
// kv cache shape = [bs, kv_head num, max_seq_len, head size] = >[bs, kv_head num, seqlen[history_len:history_len+seqlen] , head size]
// ksrc shape = [bs, kv_head num, max_q_len, head size],为什么是q_len?
// void launchAppendKVCache(Tensor*     k_src, // from qkv bias and rope
//                          Tensor*     v_src,
//                          Tensor*     layer_id,// layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
//                          Tensor*     cur_query_length, // current epoch or local input length,[batchsize]
//                          Tensor*     history_length,
//                          Tensor*     k_dst, 
//                          Tensor*     v_dst)
template<typename T>
void launchAppendKVCache(TensorWrapper<T>*     k_src, // from qkv bias and rope
                         TensorWrapper<T>*     v_src,
                         TensorWrapper<int>*     layer_id,// layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                         TensorWrapper<int>*     cur_query_length, // current epoch or local input length,[batchsize]
                         TensorWrapper<int>*     history_length,
                         TensorWrapper<T>*     k_dst, 
                         TensorWrapper<T>*     v_dst)
{
    int batch_size =k_src->shape[0];    
    int max_seq_len = k_dst->shape[2];
    int kv_head_num = k_src->shape[1];
    int max_q_len = k_src->shape[2];
    int head_size = k_src->shape[3];
    int blockSize = head_size;
    // std::cout <<"before getval"<<"\n";
    // std::cout <<layer_id->DeviceString()<<"\n";
    int layer = layer_id->getVal();
    // std::cout <<"after getval"<<"\n";
    size_t layer_offset = layer * batch_size * kv_head_num * max_seq_len * head_size;
    //note: this is for vectorization of kv cache for attention
    //constexpr int x = (sizeof(T) == 4) ? 4 : 8;
    dim3 grid(max_q_len, batch_size, kv_head_num);
    // std::cout << "calling concat kv cache kernel" << "\n";
    append_key_cache<T><<<grid, blockSize>>>(k_dst->data,
                                              layer_offset,
                                              k_src->data,
                                              kv_head_num,
                                              head_size,
                                              /*(int*)*/cur_query_length->data,
                                              /*(int*)*/history_length->data,
                                              max_q_len,
                                              max_seq_len);

    append_value_cache<T><<<grid, blockSize>>>(v_dst->data,
                                                layer_offset,
                                                v_src->data,
                                                kv_head_num,
                                                head_size,
                                                /*(int*)*/cur_query_length->data,
                                                /*(int*)*/history_length->data,
                                                max_q_len,
                                                max_seq_len);
    
    // std::cout << "called concat kv cache kernel" << "\n";

}

template void launchAppendKVCache(TensorWrapper<float>*     k_src, // from qkv bias and rope
                                TensorWrapper<float>*     v_src,
                                TensorWrapper<int>*     layer_id,// layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                                TensorWrapper<int>*     cur_query_length, // current epoch or local input length,[batchsize]
                                TensorWrapper<int>*     history_length,
                                TensorWrapper<float>*     k_dst, 
                                TensorWrapper<float>*     v_dst);

template void launchAppendKVCache(TensorWrapper<half>*     k_src, // from qkv bias and rope
                                TensorWrapper<half>*     v_src,
                                TensorWrapper<int>*     layer_id,// layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                                TensorWrapper<int>*     cur_query_length, // current epoch or local input length,[batchsize]
                                TensorWrapper<int>*     history_length,
                                TensorWrapper<half>*     k_dst, 
                                TensorWrapper<half>*     v_dst);
