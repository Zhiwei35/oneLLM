#include <iostream>
#include "src/layers/ffn/ffn.h"
#include "src/utils/macro.h"

LLaMAFFNLayer::LLaMAFFNLayer(int head_num,
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
void LLaMAFFNLayer::allocForForward(LLaMAAttentionDynParams& params){
    int num_tokens = params.num_tokens;
    DataType type = getTensorType<T>(); 
    SwiGLU_input = new Tensor(Device::GPU, type, {2, num_tokens, inter_size});
    down_proj_input = new Tensor(Device::GPU, type, {num_tokens, inter_size});
    down_proj_output = new Tensor(Device::GPU, type, {num_tokens, hidden_units});
    SwiGLU_input->data = allocator->Malloc(SwiGLU_input->data, sizeof(T) * num_tokens * 2 * inter_size, false);
    down_proj_input->data = allocator->Malloc(down_proj_input->data, sizeof(T) * num_tokens * inter_size, false);
    down_proj_output->data = allocator->Malloc(down_proj_output->data, sizeof(T) * num_tokens * hidden_units, false);
}

template<typename T>
void LLaMAFFNLayer::allocForForward(int batch_size){
    DataType type = getTensorType<T>(); 
    SwiGLU_input = new Tensor(Device::GPU, type, {2, batch_size, inter_size});
    down_proj_input = new Tensor(Device::GPU, type, {batch_size, inter_size});
    down_proj_output = new Tensor(Device::GPU, type, {batch_size, hidden_units});
    SwiGLU_input->data = allocator->Malloc(SwiGLU_input->data, sizeof(T) * batch_size * 2 * inter_size, false);
    down_proj_input->data = allocator->Malloc(down_proj_input->data, sizeof(T) * batch_size * inter_size, false);
    down_proj_output->data = allocator->Malloc(down_proj_output->data, sizeof(T) * batch_size * hidden_units, false);
}

void LLaMAFFNLayer::free(){
    allocator->deviceFree(SwiGLU_input->data);
    DeviceSyncAndCheckCudaError();
    allocator->deviceFree(down_proj_input->data);
    DeviceSyncAndCheckCudaError();
    allocator->deviceFree(down_proj_output->data);
    DeviceSyncAndCheckCudaError();
}

void LLaMAFFNLayer::forward(TensorMap& inputs, TensorMap& outputs, LLaMAFFNWeights& weights, LLaMAAttentionDynParams& params){
    if (params.num_tokens > 0) {
        allocForForward<float>(params);
    } else {
        allocForForward<float>(params.batch_size);
    }
    Tensor ffn_input = inputs["ffn_input"];
    Tensor ffn_output = outputs["ffn_output"];
    // gate proj
    launchLinearGemm(&ffn_input, weights.gate, SwiGLU_input);
    // up proj
    launchLinearGemm(&ffn_input, weights.up, SwiGLU_input, false, false, true);

    launchAct((float*)SwiGLU_input->data, (float*)down_proj_input->data, params.batch_size, weights.gate.shape[1]);// down_proj_input maybe can reuse swiglu_input buf, will validate it later
    //down proj
    launchLinearGemm(down_proj_input, weights.down, down_proj_output);

    if (is_free_buffer_after_fwd) {
        this->free();
    }
}
