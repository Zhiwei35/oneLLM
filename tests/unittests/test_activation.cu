#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cuda.h>
// #include <cuda_runtime.h>
#include "src/kernels/activation_kernel.h"

void CPUSwiGLU(float* input, float* output, int batch_size, int intermedia_size){
    for(int batch_id = 0; batch_id < batch_size; batch_id++){
        for(int i = 0; i < intermedia_size; i++) {
            int offset1 = batch_id * 2 * intermedia_size + i;
            int offset2 = batch_id * 2 * intermedia_size + i + intermedia_size;
            int out_offset = batch_id * intermedia_size + i;
            float silu_out = input[offset1] / (1.0f + expf(-input[offset1]));
            output[out_offset] = silu_out * input[offset2];
        }
    }
}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
    for(int i = 0; i < output_size; i++) {
        if(fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6){
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }

    }
    return true;
}
int main() {
    constexpr int batch_size = 16;
    constexpr int intermedia_size = 11008;
    constexpr int input_size = batch_size * intermedia_size * 2;
    constexpr int output_size = batch_size * intermedia_size;
    float* h_input;
    float* d_input;
    h_input = (float*)malloc(sizeof(float) * input_size);
    cudaMalloc((void**)&d_input, sizeof(float) * input_size);
    float* h_output;
    float* d_output;
    h_output = (float*)malloc(sizeof(float) * output_size);
    cudaMalloc((void**)&d_output, sizeof(float) * output_size);
    for(int i = 0; i < input_size; i++) { // initialize host data
        h_input[i] = (float)i;
    }
    cudaMemcpy(d_input, h_input, sizeof(float) * input_size, cudaMemcpyHostToDevice);
    launchAct(d_input, d_output, batch_size, intermedia_size);
    cudaMemcpy(h_output, d_output, sizeof(float) * output_size, cudaMemcpyDeviceToHost);
    float* CPU_output = (float*)malloc(sizeof(float) * output_size);
    CPUSwiGLU(h_input, CPU_output, batch_size, intermedia_size);
    bool is_true = CheckResult(CPU_output, h_output, output_size);
    if(is_true){
        printf("test passed");
    } else {
        printf("test failed");
    }

    free(h_input);
    free(h_output);
    free(CPU_output);
    cudaFree(d_input);
    cudaFree(d_output);
}
