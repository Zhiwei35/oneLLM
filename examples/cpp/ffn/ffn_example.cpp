#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/layers/ffn/ffn.h"
#include "src/utils/macro.h"

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

int main(int argc, char** argv)
{
    int head_num = 4;
    int head_size = 8;
    int inter_size = 12;
    int hidden_units = head_num * head_size;
    bool is_free_buffer_after_fwd = true;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    BaseAllocator* allocator = new CudaAllocator;

    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.num_tokens = 14;  
    std::cout << "start malloc/cudamalloc buffer" << "\n";
    float* h_ffn_input = (float*) malloc(sizeof(float) * hidden_units * attn_dyn_params.num_tokens);
    float* d_ffn_input;
    cudaMalloc((void**)&d_ffn_input, sizeof(float) * hidden_units * attn_dyn_params.num_tokens);
    for(int i = 0; i < hidden_units * attn_dyn_params.num_tokens; i++) { 
       h_ffn_input[i] = 1.0f;
    }    
    float* h_gate = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    float* d_gate;
    cudaMalloc((void**)&d_gate, sizeof(float) * hidden_units * inter_size);
    for(int i = 0; i < hidden_units * inter_size; i++) { 
       h_gate[i] = 1.0f;
    }  
    float* h_up = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    float* d_up;
    cudaMalloc((void**)&d_up, sizeof(float) * hidden_units * inter_size);
    for(int i = 0; i < hidden_units * inter_size; i++) { 
       h_up[i] = 1.0f;
    }  
    float* h_down = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    float* d_down;
    cudaMalloc((void**)&d_down, sizeof(float) * hidden_units * inter_size);
    for(int i = 0; i < hidden_units * inter_size; i++) { 
       h_down[i] = 1.0f;
    }  
    float* d_ffn_output;
    cudaMalloc((void**)&d_ffn_output, sizeof(float) * attn_dyn_params.num_tokens * hidden_units);
    std::cout << "end malloc/cudamalloc buffer and start memcpyh2d" << "\n";
    CHECK(cudaMemcpy(d_ffn_input, h_ffn_input, sizeof(float) * hidden_units * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_gate, h_gate, sizeof(float) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_up, h_up, sizeof(float) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_down, h_down, sizeof(float) * hidden_units * inter_size, cudaMemcpyHostToDevice));
    DataType type = getTensorType<float>(); // note: the type should be as a class data member!
    LLaMAFFNWeights<float> ffn_weights;
    ffn_weights.gate.data = d_gate;
    ffn_weights.gate.shape = {hidden_units, inter_size};
    ffn_weights.up.data = d_up;
    ffn_weights.up.shape = {hidden_units, inter_size};
    ffn_weights.down.data = d_down;
    ffn_weights.down.shape = {inter_size, hidden_units};

    TensorMap ffn_inputs{
        {"ffn_input", &TensorWrapper<float>(GPU, type, {attn_dyn_params.num_tokens, hidden_units}, d_ffn_input)}
    };
    TensorMap ffn_outputs{
        {"ffn_output", &TensorWrapper<float>(GPU, type, {attn_dyn_params.num_tokens, hidden_units}, d_ffn_output)}
    };
    std::cout << "initializing ffn layer" << "\n";
    LLaMAFFNLayer<float>* ffn_layer = new LLaMAFFNLayer<float>(head_num,
                                                head_size,
                                                inter_size,
                                                stream,
                                                cublas_wrapper,
                                                allocator,
                                                is_free_buffer_after_fwd);
    std::cout << "start fwd" << "\n";
    ffn_layer->forward(ffn_inputs, ffn_outputs, ffn_weights, attn_dyn_params);
    std::cout << "end fwd" << "\n";
    free(h_ffn_input);  
    free(h_gate);  
    free(h_up);  
    free(h_down); 
    cudaFree(d_ffn_input);  
    cudaFree(d_gate);  
    cudaFree(d_up);  
    cudaFree(d_down); 
    cudaFree(d_ffn_output);
}
