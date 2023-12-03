#include "src/weights/llama/attention_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/qkv_linear.h"
#include "src/kernels/attn_softmax_kernel.h"
#include "src/kernels/qkv_bias_and_RoPE.h"
#include "src/kernels/fused_transpose_reshape_remv_pad.h"
#include "src/kernels/append_to_kvcache.h"
#include "src/kernels/transpose_kernel.h"
#include "src/utils/tensor.h"
#include "src/kernels/cublas_wrapper.h"
#include "src/models/llama/llama_params.h"
template<typename T>
class LLaMAContextAttentionLayer {
private:
    // this params are shared across all LLMs
    const int head_num;
    const int head_size;
    const int hidden_units;
    const int q_head_per_kv; //for GQA and MQA
    const int kv_head_num;
    const bool is_free_buffer_after_fwd;
    //const bool is_1st_epoch; // judge if its 1st epoch, if so, we will allocate kv cache
    float scale;
    // this params are only saw in llama and are unchanged 
    LLaMAAttentionStaticParams attn_static_params;
    // this params are dynamic
    //const LLaMAAttentionDynParams attn_dyn_params;

    cudaStream_t stream;
    BaseAllocator* allocator;
    // for linear and batchgemm
    cublasWrapper* cublas_wrapper;

    // buffer
    // T*     qkv_buf_wo_pad = nullptr;
    // T*     q_buf_w_pad = nullptr;
    // T*     k_buf_w_pad = nullptr;
    // T*     v_buf_w_pad = nullptr;
    // T*     k_cache_buf = nullptr;
    // T*     v_cache_buf = nullptr;
    // T*     qk_buf = nullptr;
    // float* qk_buf_float = nullptr; // for acc
    // T*     qkv_buf_w_pad = nullptr;

    TensorWrapper<T>*  qkv_buf_wo_pad = nullptr;      
    TensorWrapper<T>*  q_buf_w_pad = nullptr;
    TensorWrapper<T>*  k_buf_w_pad = nullptr;
    TensorWrapper<T>*  v_buf_w_pad = nullptr;
    TensorWrapper<T>*  k_cache_buf = nullptr;
    TensorWrapper<T>*  v_cache_buf = nullptr;
    TensorWrapper<T>*  qk_buf = nullptr;
    TensorWrapper<T>*  qk_buf_float = nullptr; // for acc
    TensorWrapper<T>*  qkv_buf_w_pad = nullptr;
    TensorWrapper<T>*  qkv_buf_wo_pad_1 = nullptr;      


public:
    LLaMAContextAttentionLayer(int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator,
                               bool is_free_buffer_after_fwd);
    LLaMAAttentionStaticParams& GetAttnStaticParams(){
        return attn_static_params;
    }
    
    void allocForForward(LLaMAAttentionDynParams& params);
    void free();
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights, LLaMAAttentionDynParams& params, LLaMAAttentionStaticParams& static_params);
    // void naiveMHA(float*          key_cache_ptr,
    //               float*          val_cache_ptr,
    //               size_t       cache_layer_offset,
    //               const float*     attention_mask,
    //               const int*   padding_offset,
    //               const int*   context_length,
    //               int          batch_size,
    //               int          num_tokens,
    //               int          max_q_len,
    //               int          max_k_len,
    //               int          max_seq_len); 
                  // whats the diff across these 3 max len:
                  // max_seq_len is the max kv len considering context, ep. multiple epochs chat
                  // max_q_len is the current max q len after padding
                  // I dont think max_k_len is the current max k len after padding, that is, the max kv cache lenght at current epoch. because in transpose kv cache
                    // the max k len is used to take context length, which is < max seqlen, so I think, the max k len is max seq len   
    void flashAttn();
};
