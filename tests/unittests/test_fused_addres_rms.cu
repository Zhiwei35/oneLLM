#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <iostream>
#include "src/kernels/fused_addresidual_norm.h"

#include <stdio.h>

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

void CPUfusedresidandRMSNorm(float* h_residual, float* h_decoder_out, float* h_bias, 
                                    float* h_scale, float eps, int hidden_units, int num_tokens) {
    for(int b = 0; b < num_tokens; b++) {
        float inv_fenmu = 0.0f;
        float mean = 0.0f;
        for (int i = 0; i < hidden_units; i++) {
            h_decoder_out[b * hidden_units + i] +=
                    h_residual[b * hidden_units + i] + h_bias[i];
        }
        for (int i = 0; i < hidden_units; i++) {
            sum += h_decoder_out[b * hidden_units + i] * h_decoder_out[b * hidden_units + i];
        }
        mean = sum / hidden_units;
        inv_fenmu = rsqrt(mean + eps);
        for (int i = 0; i < hidden_units; i++) {
            h_decoder_out[b * hidden_units + i] = h_decoder_out[b * hidden_units + i] * inv_fenmu * h_scale[i];
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
    const int num_tokens = 2;
    const int hidden_units = 32;
    const int total_size = num_tokens * hidden_units;
    float eps = 0.5f;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_residual;
    float* d_residual;
    h_residual = (float*)malloc(sizeof(float) * total_size);
    cudaMalloc((void**)&d_residual, sizeof(float) * total_size);
    for(int i = 0; i < total_size; i++) { 
       h_residual[i] = 1.0f;
    }

    float* h_decoder_out = (float*) malloc(sizeof(float) * total_size);
    float* d_decoder_out;
    cudaMalloc((void**)&d_decoder_out, sizeof(float) * total_size);
    for(int i = 0; i < total_size; i++) { 
       h_decoder_out[i] = 1.0f;
    }
    //bias
    float* h_bias = (float*) malloc(sizeof(float) * hidden_units);
    float* d_bias;
    cudaMalloc((void**)&d_bias, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units; i++) { 
       h_bias[i] = 1.0f;
    }
    //rmsnorm weights
    float* h_scale = (float*) malloc(sizeof(float) * hidden_units);
    float* d_scale;
    cudaMalloc((void**)&d_scale, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units; i++) { 
       h_scale[i] = 1.0f;
    }

    CHECK(cudaMemcpy(d_residual, h_residual, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_decoder_out, h_decoder_out, sizeof(float) * total_size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_bias, h_bias, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_scale, h_scale, sizeof(float) * hidden_units, cudaMemcpyHostToDevice));

    // debug info, better to retain: 
    std::cout << "before launch kernel" << std::endl;
    launchFusedAddBiasResidualRMSNorm(d_residual, 
                                    d_decoder_out, 
                                    d_bias,
                                    d_scale,
                                    eps,
                                    num_tokens,
                                    hidden_units);
    // debug info, better to retain: 
    std::cout << "after launch kernel" << std::endl;
    // debug info, better to retain: 
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    CHECK(cudaMemcpy(h_decoder_out, d_decoder_out, sizeof(float) * total_size, cudaMemcpyDeviceToHost));
    //cublasGetVector(hidden_units, sizeof(float), d_out, 1, h_out, 1);
    float* CPUout = (float*) malloc(sizeof(float) * total_size);
    CPUfusedresidandRMSNorm(h_residual, h_decoder_out, h_bias, 
                h_scale, eps, hidden_units, num_tokens);
    bool is_right = CheckResult(CPUout, h_decoder_out, total_size);
    // debug info, better to retain: 
    std::cout << "before free" << std::endl;
    std::cout << "linear passed" << std::endl;
    free(h_residual);
    free(h_decoder_out);
    free(h_bias);
    free(h_scale);
    free(CPUout);
    cudaFree(d_residual);
    cudaFree(d_decoder_out);
    cudaFree(d_bias);
    cudaFree(d_scale);
}