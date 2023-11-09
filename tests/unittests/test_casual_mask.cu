#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include "src/kernels/build_casual_mask.h"
void CPUbuildCasualMask(float* mask, 
                        const int* q_lens,  //input lens, shape=[batch size]
                        const int* k_lens,  //context lens, shape=[batch size]
                        int max_q_len, 
                        int max_k_len,
                        int batch_size) {
    for(int b = 0; b < batch_size; b++){
        int start = b * max_q_len * max_k_len;
        int q = q_lens[b];
        int k = k_lens[b];
        for(int i = 0; i < max_q_len; i++) {
            for(int j = 0; j < max_k_len; j++) {
                if(j <= i + (k - q) && i < q && j < k) {
                    mask[start + i * max_k_len + j] = static_cast<float>(1);
                }
            }
        }
    }
}
bool CheckResult(float* CPUres, float* GPUres, const int size) {
    for(int i = 0; i < size; i++) {
        if(fabs(CPUres[i] - GPUres[i]) > 1e-6){
            printf("the %dth res is wrong, CPU mask = %f, GPU mask = %f\n", i, CPUres[i], GPUres[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int batch_size = 1;
    const int max_q_len = 5;
    const int max_k_len = 5;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int mask_size = batch_size * max_q_len * max_k_len;
    int* h_q_lens;
    int* d_q_lens;
    h_q_lens = (int*)malloc(sizeof(int) * max_q_len);
    cudaMalloc((void**)&d_q_lens, sizeof(int) * max_q_len);
    int* h_k_lens;
    int* d_k_lens;
    h_k_lens = (int*)malloc(sizeof(int) * max_k_len);
    cudaMalloc((void**)&d_k_lens, sizeof(int) * max_k_len);

    float* d_mask;
    float* h_mask = (float*)malloc(sizeof(float) * mask_size);
    cudaMalloc((void**)&d_mask, sizeof(float) * mask_size);

    for(int i = 0; i < max_q_len; i++) {
       h_q_lens[i] = 3;
    }
    for(int i = 0; i < max_k_len; i++) {
       h_k_lens[i] = 3;
    }
    cudaMemcpy(d_q_lens, h_q_lens, sizeof(int) * max_q_len, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_lens, h_k_lens, sizeof(int) * max_k_len, cudaMemcpyHostToDevice);
    // debug info, better to retain: std::cout << "before launch kernel" << std::endl;
    launchBuildCausalMasks(d_mask, d_q_lens, d_k_lens, max_q_len, max_k_len);
    // debug info, better to retain: std::cout << "after launch kernel" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_mask, d_mask, sizeof(float) * mask_size, cudaMemcpyDeviceToHost);
    CPUbuildCasualMask(CPUmask, h_q_lens, h_k_lens, max_q_len, max_k_len, batch_size);
    if (CheckResult(CPUmask, h_mask, mask_size)) {
        printf("test passed!\n");
    }

    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_q_lens);
    free(h_k_lens);
    free(h_mask);
    cudaFree(d_q_lens);
    cudaFree(d_k_lens);
    cudaFree(d_mask);
}
