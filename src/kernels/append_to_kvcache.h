#include <cuda_runtime.h>
#include <cuda.h>
#include "src/utils/tensor.h"
// void launchAppendKVCache(T*          k_dst, // 每个layer都有单独的kv cache
//                          T*          v_dst, // 猜测为二级指针的原因是每个layer都单独一份，所以每个一级指针为每个layer的kv cache
//                          int       layer_offset,//layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
//                          const T*     k_src, // from qkv bias and rope
//                          const T*     v_src,
//                          int          local_batch_size, // local bs may mean the bs in the current chat epoch
//                          const int*   cur_query_length, // current epoch or local input length,[batchsize]
//                          int          max_q_len, //query 的最大长度(after padding)
//                          const int*   history_length,
//                          int          max_seq_len, // kv cache的最大长度
//                          int          head_size,
//                          int          local_head_num,
// //                         cudaStream_t stream,
//                          bool          quant,
//                          const float* kv_scale);

void launchAppendKVCache(Tensor*     k_src, // from qkv bias and rope
                         Tensor*     v_src,
                         Tensor*     layer_id,// layer offset = layer_id * batchxbeam * max_seq_len * kv_head_num * head_size
                         Tensor*     cur_query_length, // current epoch or local input length,[batchsize]
                         Tensor*     history_length,
                         Tensor*     k_dst, 
                         Tensor*     v_dst);//少写一个;都会发生很多奇怪的错误
