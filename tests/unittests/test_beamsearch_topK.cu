#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cuda.h>
// #include <cuda_runtime.h>
#include "src/kernels/beamsearch_topK.h"

int main() {
    const int batch_size = 1;
    const int vocab_size = 30000;
    const int beamwidth = 2;
    std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int probs_size = batch_size * vocab_size * beamwidth;
    float* h_probs;
    float *d_probs;
    h_probs = (float*)malloc(sizeof(float) * probs_size);
    cudaMalloc((void**)&d_probs, sizeof(float) * probs_size);
    float* topK_workspace;
    cudaMalloc((void**)&topK_workspace, sizeof(float) * (2 * batch_size * beamwidth + 2 * batch_size * beamwidth * 8/*max block per beam*/ * beamwidth));
    for(int i = 0; i < probs_size; i++) { // 0-59999
        // h_probs[i] = rand() % 100 / (float)(100 + 1);
        // if (h_probs[i] > 1 || h_probs[i] < 0) {
        //     std::cout << "warning!! probs exceed [0,1]" << std::endl;
        // }
        h_probs[i] = i;
    }
    cudaMemcpy(d_probs, h_probs, sizeof(float)*probs_size, cudaMemcpyHostToDevice);
    std::cout << "before launch kernel" << std::endl;
    launchTopKforBeamSearch(d_probs, batch_size, vocab_size, topK_workspace);
    std::cout << "after launch kernel" << std::endl;
    int* h_topK_workspace = (int*)malloc(sizeof(int) * (batch_size * beamwidth));
    std::cout << "cuda memcpy device to host" << std::endl;
    cudaMemcpy(h_topK_workspace, topK_workspace + batch_size * beamwidth + 2 * batch_size * beamwidth * 8 * beamwidth, sizeof(int) * batch_size * beamwidth, cudaMemcpyDeviceToHost);
    //int* res = (int*) (topK_workspace + batch_size * beamwidth + 2 * batch_size * beamwidth * 8/*max block per beam*/ * beamwidth);
    for(int i = 0; i < beamwidth; i++) {
        int id = h_topK_workspace[i];
        printf("topK id = %d\n", id);
    }
    std::cout << "before free" << std::endl;
    free(h_probs);
    free(h_topK_workspace);
    cudaFree(d_probs);
    cudaFree(topK_workspace);
}
