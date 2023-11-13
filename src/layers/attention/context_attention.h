#include "src/weights/llama/attention_weights.h"
#include "src/kernels/cublas_wrapper.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/utils/tensor.h"

struct LLaMAAttentionStaticParams{
    int   rotray_embedding_dim;
    float rotary_embedding_base;
    int   max_position_embeddings;
    bool  use_dynamic_ntk; // for dyn scaling rope
};

struct LLaMAAttentionDynParams {
    int batch_size;
    int num_tokens;
    int max_q_len;
    int max_k_len;
    int num_layers;
};

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
    const bool is_1st_epoch; // judge if its 1st epoch, if so, we will allocate kv cache

    // this params are only saw in llama and are unchanged 
    const LLaMAAttentionStaticParams attn_static_params;
    // this params are dynamic
    const LLaMAAttentionDynParams attn_dyn_params;

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

    Tensor*  qkv_buf_wo_pad = nullptr;      
    Tensor*  q_buf_w_pad = nullptr;
    Tensor*  k_buf_w_pad = nullptr;
    Tensor*  v_buf_w_pad = nullptr;
    Tensor*  k_cache_buf = nullptr;
    Tensor*  v_cache_buf = nullptr;
    Tensor*  qk_buf = nullptr;
    Tensor*  qk_buf_float = nullptr; // for acc
    Tensor*  qkv_buf_w_pad = nullptr;
    Tensor*  qkv_buf_wo_pad_1 = nullptr;      


public:
    LLaMAContextAttentionLayer(int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator,
                               bool is_free_buffer_after_fwd);
    
    void allocForForward(LLaMAAttentionDynParams params);
    void free();
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights);
    void naiveMHA(T*          key_cache_ptr,
                  T*          val_cache_ptr,
                  size_t       cache_layer_offset,
                  const T*     attention_mask,
                  const int*   padding_offset,
                  const int*   context_length,
                  int          batch_size,
                  int          num_tokens,
                  int          max_q_len,
                  int          max_k_len,
                  int          max_seq_len); // whats the diff across these 3 max len
        
    void flashAttn();
};