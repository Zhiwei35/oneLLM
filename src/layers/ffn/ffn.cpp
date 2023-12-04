#include <iostream>
#include "src/layers/ffn/ffn.h"
#include "src/utils/macro.h"
template<typename T>
LLaMAFFNLayer<T>::LLaMAFFNLayer(int head_num,
                               int head_size,
                               int inter_size,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator,
                               bool is_free_buffer_after_fwd):
    head_num(head_num),
    head_size(head_size),
    inter_size(inter_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator), //cudaAllocator
    hidden_units(head_num * head_size),
    is_free_buffer_after_fwd(is_free_buffer_after_fwd) {}
template<typename T>
void LLaMAFFNLayer<T>::allocForForward(LLaMAAttentionDynParams& params){
    int num_tokens = params.num_tokens;
    DataType type = getTensorType<T>(); 
    SwiGLU_input = new TensorWrapper<T>(Device::GPU, type, {2, num_tokens, inter_size});
    down_proj_input = new TensorWrapper<T>(Device::GPU, type, {num_tokens, inter_size});
    // down_proj_output = new TensorWrapper<T>(Device::GPU, type, {num_tokens, hidden_units});
    SwiGLU_input->data = allocator->Malloc(SwiGLU_input->data, sizeof(T) * num_tokens * 2 * inter_size, false);
    down_proj_input->data = allocator->Malloc(down_proj_input->data, sizeof(T) * num_tokens * inter_size, false);
    // down_proj_output->data = allocator->Malloc(down_proj_output->data, sizeof(T) * num_tokens * hidden_units, false);
}
template<typename T>
void LLaMAFFNLayer<T>::allocForForward(int batch_size){
    DataType type = getTensorType<T>(); 
    SwiGLU_input = new TensorWrapper<T>(Device::GPU, type, {2, batch_size, inter_size});
    down_proj_input = new TensorWrapper<T>(Device::GPU, type, {batch_size, inter_size});
    // down_proj_output = new TensorWrapper<T>(Device::GPU, type, {batch_size, hidden_units});
    SwiGLU_input->data = allocator->Malloc(SwiGLU_input->data, sizeof(T) * batch_size * 2 * inter_size, false);
    down_proj_input->data = allocator->Malloc(down_proj_input->data, sizeof(T) * batch_size * inter_size, false);
    // down_proj_output->data = allocator->Malloc(down_proj_output->data, sizeof(T) * batch_size * hidden_units, false);
}
template<typename T>
void LLaMAFFNLayer<T>::freeBuf(){
    allocator->Free(SwiGLU_input->data);
    DeviceSyncAndCheckCudaError();
    allocator->Free(down_proj_input->data);
    DeviceSyncAndCheckCudaError();
    // allocator->Free(down_proj_output->data);
    // DeviceSyncAndCheckCudaError();
}
template<typename T>
void LLaMAFFNLayer<T>::forward(TensorMap& inputs, TensorMap& outputs, LLaMAFFNWeights<T>& weights, LLaMAAttentionDynParams& params){
    if (params.num_tokens > 0) {
        allocForForward<T>(params);
    } else {
        allocForForward<T>(params.batch_size);
    }
    Tensor* ffn_input = inputs["ffn_input"];
    Tensor* ffn_output = outputs["ffn_output"];
    // gate proj
    launchLinearGemm(ffn_input->as<T>(), weights.gate, SwiGLU_input, cublas_wrapper);
    // up proj
    launchLinearGemm(ffn_input->as<T>(), weights.up, SwiGLU_input, cublas_wrapper, false, false, true);

    launchAct(SwiGLU_input->as<T>(), down_proj_input->as<T>(), params.batch_size, weights.gate.shape[1]);// down_proj_input maybe can reuse swiglu_input buf, will validate it later
    //down proj
    // error, output should be ffn output
    // launchLinearGemm(down_proj_input, weights.down, down_proj_output);
    launchLinearGemm(down_proj_input, weights.down, ffn_output->as<T>(), cublas_wrapper);

    if (is_free_buffer_after_fwd) {
        this->freeBuf();
    }
};
