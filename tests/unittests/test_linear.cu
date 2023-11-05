#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cuda.h>
#include <iostream.h>
#include "src/kernels/qkv_linear.h"

void CPUlinear(float* input, float* weight, float* output,
                int m, int k, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int l = 0; l < k; l++) {
                output[i * n + j] = input[i * k + l] * weight[l * n + j];
            }
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
    const int seqlen = 1;
    const int hidden_units = 16;
    const int hidden_units_2 = 256
    // (16, 16) * (16, 1)
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_w;
    float* d_w;
    h_w = (float*)malloc(sizeof(float) * hidden_units_2);
    cudaMalloc((void**)&d_w, sizeof(float) * hidden_units_2);
    for(int i = 0; i < hidden_units_2; i++) { 
       h_w[i] = 1.0f;
    }

    float* h_in = (float*) malloc(sizeof(float) * hidden_units);
    float* d_in;
    cudaMalloc((void**)&d_in, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units; i++) { 
       h_in[i] = 1.0f;
    }

    float* h_out = (float*) malloc(sizeof(float) * hidden_units);
    float* d_out;
    cudaMalloc((void**)&d_out, sizeof(float) * hidden_units);

    cudaMemcpy(d_in, h_in, sizeof(float) * hidden_units_2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w, sizeof(float) * hidden_units_2, cudaMemcpyHostToDevice);
    // debug info, better to retain: 
    std::cout << "before launch kernel" << std::endl;
    launchLinear<float>(d_in, d_out, seqlen, d_w, hidden_units);
    // debug info, better to retain: 
    std::cout << "after launch kernel" << std::endl;
    // debug info, better to retain: 
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_out, d_out, sizeof(float) * hidden_units, cudaMemcpyDeviceToHost);
    float* CPUout = (float*) malloc(sizeof(float) * hidden_units);
    CPUlinear(h_in, h_w, CPUout, hidden_units, hidden_units, seqlen);
    bool is_right = CheckResult(CPUout, h_out, hidden_units);
    // debug info, better to retain: 
    std::cout << "before free" << std::endl;
    std::cout << "passed" << std::endl;
    free(h_in);
    free(h_w);
    free(h_out);
    free(CPUout);
    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_out);
}