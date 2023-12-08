#pragma once
#include "src/weights/llama/attention_weights.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/kernels/qkv_linear.h"
#include "src/utils/tensor.h"
#include "src/kernels/cublas_wrapper.h"
#include "src/models/llama/llama_params.h"
#include "src/kernels/activation_kernel.h"
template<typename T>
class LLaMAFFNLayer {
private:
    // this params are shared across all LLMs
    const int head_num;
    const int head_size;
    const int inter_size;
    const int hidden_units;
    const bool is_free_buffer_after_fwd;
//    const bool is_1st_epoch; // judge if its 1st epoch, if so, we will allocate kv cache
    // this params are dynamic
    //const LLaMAAttentionDynParams attn_dyn_params;

    cudaStream_t stream;
    BaseAllocator* allocator;
    // for linear proj
    cublasWrapper* cublas_wrapper;

    // buffer
    // [num tokens, 2, intersize]
    TensorWrapper<T>*  SwiGLU_input = nullptr;  //gate proj and up proj output buf   
    // [num tokens, intersize] 
    TensorWrapper<T>*  down_proj_input = nullptr; 
    // [num tokens, hiddenunits]
    // TensorWrapper<T>*  down_proj_output = nullptr;
  


public:
    LLaMAFFNLayer(int head_num,
                    int head_size,
                    int inter_size,
                    cudaStream_t stream,
                    cublasWrapper* cublas_wrapper,
                    BaseAllocator* allocator,
                    bool is_free_buffer_after_fwd);

    void allocForForward(LLaMAAttentionDynParams& params);
    void allocForForward(int batch_size);
    void freeBuf();
    void forward(TensorMap& inputs, TensorMap& outputs, LLaMAFFNWeights<T>& weights, LLaMAAttentionDynParams& params);
};
