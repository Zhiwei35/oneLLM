#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cuda.h>
#include "src/kernels/beamsearch_topK.h"

int main() {
    const int batch_size = 1;
    const int vocab_size = 30000;
    const int beamwidth = 2;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int probs_size = batch_size * vocab_size * beamwidth;
    float* h_probs;
    float *d_probs;
    h_probs = (float*)malloc(sizeof(float) * probs_size);
    cudaMalloc((void**)&d_probs, sizeof(float) * probs_size);
    float* topK_workspace;
    cudaMalloc((void**)&topK_workspace, sizeof(float) * (2 * batch_size * beamwidth + 2 * batch_size * beamwidth * 8/*max block per beam*/ * beamwidth));
    for(int i = 0; i < probs_size; i++) { // 0-59999
       h_probs[i] = i;
    }
    cudaMemcpy(d_probs, h_probs, sizeof(float)*probs_size, cudaMemcpyHostToDevice);
    // debug info, better to retain: std::cout << "before launch kernel" << std::endl;
    launchTopKforBeamSearch(d_probs, batch_size, vocab_size, topK_workspace);
    // debug info, better to retain: std::cout << "after launch kernel" << std::endl;
    int* h_topK_workspace = (int*)malloc(sizeof(int) * (batch_size * beamwidth));
    // debug info, better to retain: std::cout << "cuda memcpy device to host" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_topK_workspace, topK_workspace + batch_size * beamwidth + 2 * batch_size * beamwidth * 8 * beamwidth, sizeof(int) * batch_size * beamwidth, cudaMemcpyDeviceToHost);
    float* h_topK_val = (float*)malloc(sizeof(float) * (batch_size * beamwidth));
    cudaMemcpy(h_topK_val, topK_workspace + 2 * batch_size * beamwidth * 8 * beamwidth,  sizeof(float) * batch_size * beamwidth, cudaMemcpyDeviceToHost);
    for(int i = 0; i < beamwidth; i++) {
        int id = h_topK_workspace[i];
        // debug info, better to retain: printf("topK id = %d\n", id);
        float val = h_topK_val[i];
        // debug info, better to retain: printf("topK val =%f\n", val);
    }
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_probs);
    free(h_topK_workspace);
    free(h_topK_val);
    cudaFree(d_probs);
    cudaFree(topK_workspace);
}
