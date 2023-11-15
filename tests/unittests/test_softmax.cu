#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <math.h>
#include "src/kernels/attn_softmax_kernel.h"

int main() {
    const int batch_size = 1;
    const int head_num = 2;
    const int q_length = 8;
    const int k_length = 8;
    const int head_size = 4;
    float scale = rsqrt(float(head_size));
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int qk_size = batch_size * head_num * q_length * k_length;
    float* h_qk;
    float* d_qk;
    h_qk = (float*)malloc(sizeof(float) * qk_size);
    cudaMalloc((void**)&d_qk, sizeof(float) * qk_size);
    float* h_score;
    float* d_score;
    h_score = (float*)malloc(sizeof(float) * qk_size);
    cudaMalloc((void**)&d_score, sizeof(float) * qk_size);
    uint8_t* h_mask;
    uint8_t* d_mask;
    h_mask = (uint8_t*)malloc(sizeof(uint8_t) *  batch_size * q_length * k_length);
    cudaMalloc((void**)&d_mask, sizeof(uint8_t) * batch_size * q_length * k_length);
    
    for(int i = 0; i < qk_size; i++) {
       h_qk[i] = 4.0f;
    }
    for(int i = 0; i < batch_size * q_length * k_length; i++) {
       h_mask[i] = (uint8_t)(i % 255);
    }    

    cudaMemcpy(d_qk, h_qk, sizeof(float)* qk_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeof(uint8_t)* batch_size * q_length * k_length, cudaMemcpyHostToDevice);
    DataType type = getTensorType<float>(); 
    Tensor qk(Device::GPU, type, {batch_size, head_num, q_length, k_length}, d_qk);
    Tensor mask(Device::GPU, type, {batch_size, q_length, k_length}, d_mask);
    Tensor score(Device::GPU, type, {batch_size, head_num, q_length, k_length});
    std::cout << "before launch softmax kernel" << std::endl;
    launchScaleMaskAndSoftmax(&qk, &mask, &score, scale);
    std::cout << "after launch softmax kernel" << std::endl;
    std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_score, score.data, sizeof(float) * qk_size, cudaMemcpyDeviceToHost);
    for(int i = 0; i < qk_size; i++) {
        printf("attn score[%d] = %f\n", i, h_score[i]);
    }
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_qk);
    free(h_score);
    free(h_mask);
    cudaFree(d_qk);
    cudaFree(d_score);
    cudaFree(d_mask);
}
