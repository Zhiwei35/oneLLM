// k/v shape = [bs, head num, max_q_len, head size] // 为什么这里不是max_k_len，新进来的kv应该是max_k_len和max_v_len
// kv cache shape = [bs, head num, max_seq_len, head size] = >[bs, head num, seqlen[history_len:history_len+seqlen] , head size]
// kv cache 是每个layer都有单独的kv cache ， from llama_from_ft.cc#104
// ksrc shape = [bs, head num, max_q_seqlen, head size],为什么是q_len?

// local head num here is kv head num
#include "src/kernels/append_to_kvcache.h"

template<typename T>
__global__ void append_key_cache(T*          k_dst, //[num layers, bs, head num, max_q_len, head size]
                                 const size_t layer_offset,
                                 const T*     k_src,
                                 const int    head_num,
                                 const int    head_size,
                                 const int*   cur_query_length,
                                 const int*   history_length,
                                 const int    max_q_len, 
                                 const int    max_seq_len)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tid = threadIdx.x;
    int token_id = (blockIdx.x * blockDim.x + tid) / head_size;
    
    // 当前layer的k cache
    T* k_cache_dst = k_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id];
    //note: the if judge is a must, because the max_q_len is GTE than cur_seq_len.
    if(token_id < cur_seq_len){
    // [batch, head num, max_q_len, head size] -> [batch, head num, maxseqlen[cumsum_seq_len:cumsum_seq_len+cur_seq_len], head size]
        int src_offset = batch_id * head_num * max_q_len * head_size + //为什么这里不是max_k_len，新进来的kv应该是max_k_len和max_v_len
                            head_id * max_q_len * head_size + 
                                token_id * head_size + tid;
        int dst_offset = batch_id * head_num * max_seq_len * head_size +
                            head_id * max_seq_len * head_size + 
                                (cumsum_seq_len + token_id) * head_size + tid;
        k_dst[dst_offset] = k_src[src_offset];
    }
}

template<typename T>
__global__ void append_value_cache(T*          v_dst,
                                    const size_t layer_offset,
                                    const T*     v_src,
                                    const int    head_num,
                                    const int    head_size,
                                    const int*   cur_query_length,
                                    const int*   history_length,
                                    const int    max_q_len, 
                                    const int    max_seq_len)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int tid = threadIdx.x;
    int token_id = (blockIdx.x * blockDim.x + tid) / head_size;
    
    // 当前layer的v cache
    T* v_cache_dst = v_dst + layer_offset;
    int cur_seq_len = cur_query_length[batch_id];
    int cumsum_seq_len = history_length[batch_id];
    //note: the if judge is a must, because the max_q_len is GTE than cur_seq_len.
    if(token_id < cur_seq_len){
    // [batch, head num, max_q_len, head size] -> [batch, head num, maxseqlen[cumsum_seq_len:cumsum_seq_len+cur_seq_len], head size]
        int src_offset = batch_id * head_num * max_q_len * head_size + 
                            head_id * max_q_len * head_size + 
                                token_id * head_size + tid;
        int dst_offset = batch_id * head_num * max_seq_len * head_size +
                            head_id * max_seq_len * head_size + 
                                (cumsum_seq_len + token_id) * head_size + tid;
        v_dst[dst_offset] = v_src[src_offset];
    }
}

template<typename T>
void launchAppendKVCache(T*          k_dst, // 每个layer都有单独的kv cache
                         T*          v_dst, // 猜测为二级指针的原因是每个layer都单独一份，所以每个一级指针为每个layer的kv cache
                         size_t       layer_offset,//layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                         const T*     k_src, // from qkv bias and rope
                         const T*     v_src,
                         int          local_batch_size, // local bs may mean the bs in the current chat epoch
                         const int*   cur_query_length, // current epoch or local input length,[batchsize] <= max_q_len, need padding to max q len
                         int          max_q_len, //query 的最大长度(after padding)
                         const int*   history_length,
                         int          max_seq_len, // kv cache的最大长度
                         int          head_size,
                         int          local_head_num,
                         //cudaStream_t stream,
                         bool          quant,
                         const float* kv_scale) // placeholder for int8/int4 kv cache
{
    constexpr int blockSize = 128;
    //note: this is for vectorization of kv cache for attention
    //constexpr int x = (sizeof(T) == 4) ? 4 : 8;

    // dim3 grid((max_q_len * head_size / x + blockSize - 1) / blockSize, local_batch_size, local_head_num);
    dim3 grid((max_q_len * head_size + blockSize - 1) / blockSize, local_batch_size, local_head_num);

    if (quant & kv_scale != nullptr) {
    }
    else {
        append_key_cache<<<grid, blockSize>>>(k_dst,
                                              layer_offset,
                                              k_src,
                                              local_head_num,
                                              head_size,
                                              cur_query_length,
                                              history_length,
                                              max_q_len,
                                              max_seq_len);

        append_value_cache<<<grid, blockSize>>>(v_dst,
                                                layer_offset,
                                                v_src,
                                                local_head_num,
                                                head_size,
                                                cur_query_length,
                                                history_length,
                                                max_q_len,
                                                max_seq_len);
    }
}

template void launchAppendKVCache(float**          k_dst, 
                                  float**          v_dst, 
                                  size_t       layer_offset,
                                  const float*     k_src, 
                                  const float*     v_src,
                                  int          local_batch_size, 
                                  const int*   cur_query_length, 
                                  int          max_q_len, 
                                  const int*   history_length,
                                  int          max_seq_len, 
                                  int          head_size,
                                  int          local_head_num,
                                  //cudaStream_t stream,
                                  bool          quant,
                                  const float* kv_scale);