#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <iostream>
#include "src/kernels/qkv_linear.h"
#include "src/weights/base_weights.h"

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

void CPUlinear(float* input, float* weight, float* output,
                int m, int k, int n) {
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            for(int l = 0; l < k; l++) {
                output[i * n + j] += weight[i * k + l] * input[l * n + j];
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
    const int hidden_units_2 = 256;
    // (16, 16) * (16, 1)
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    float* h_w;
    float* d_w;
    h_w = (float*)malloc(sizeof(float) * hidden_units_2);
    cudaMalloc((void**)&d_w, sizeof(float) * hidden_units_2);
    for(int i = 0; i < hidden_units_2; i++) { 
       h_w[i] = 1.0f;
    }

    float* h_in = (float*) malloc(sizeof(float) * hidden_units * seqlen);
    float* d_in;
    cudaMalloc((void**)&d_in, sizeof(float) * seqlen *  hidden_units);
    for(int i = 0; i < hidden_units * seqlen; i++) { 
       h_in[i] = 1.0f;
    }

    float* h_out = (float*) malloc(sizeof(float) * hidden_units * seqlen);
    float* d_out;
    cudaMalloc((void**)&d_out, sizeof(float) * hidden_units * seqlen);

    CHECK(cudaMemcpy(d_in, h_in, sizeof(float) * hidden_units * seqlen, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_w, h_w, sizeof(float) * hidden_units_2, cudaMemcpyHostToDevice));
    DataType type = getTensorType<float>();
    WeightType wtype = getWeightType<float>(); 
    Tensor in(Device::GPU, type, {seqlen, hidden_units}, d_in);
    BaseWeight weight;
    weight.shape = {hidden_units, hidden_units};
    weight.data = d_w;
    weight.type = wtype;
    Tensor out(Device::GPU, type, {seqlen, hidden_units}, d_out);
    // debug info, better to retain: 
    std::cout << "before launch kernel" << std::endl;
    launchLinearGemm(&in, weight, &out);
    // debug info, better to retain: 
    std::cout << "after launch kernel" << std::endl;
    // debug info, better to retain: 
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    CHECK(cudaMemcpy(h_out, out.data, sizeof(float) * hidden_units * seqlen, cudaMemcpyDeviceToHost));
    //cublasGetVector(hidden_units, sizeof(float), d_out, 1, h_out, 1);
    float* CPUout = (float*) malloc(sizeof(float) * hidden_units * seqlen);
    CPUlinear(h_in, h_w, CPUout, hidden_units, hidden_units, seqlen);
    bool is_right = CheckResult(CPUout, h_out, hidden_units * seqlen);
    // debug info, better to retain: 
    std::cout << "before free" << std::endl;
    std::cout << "linear passed" << std::endl;
    free(h_in);
    free(h_w);
    free(h_out);
    free(CPUout);
    cudaFree(d_in);
    cudaFree(d_w);
    cudaFree(d_out);
}
