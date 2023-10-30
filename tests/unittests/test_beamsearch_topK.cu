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
    constexpr int batch_size = 1;
    constexpr int vocab_size = 30000;
    constexpr int beamwidth = 2;
    constexpr int probs_size = batch_size * vocab_size * beamwidth;
    float* h_probs;
    float *d_probs;
    h_probs = (float*)malloc(sizeof(float) * probs_size);
    cudaMalloc((void**)&d_probs, sizeof(float) * probs_size);
    float* topK_workspace;
    cudaMalloc((void**)&topK_workspace, 4 * batch_size * beamwidth * 8/*max block per beam*/ * beamwidth);
    for(int i = 0; i < probs_size; i++) { // 0-59999
        // h_probs[i] = rand() % 100 / (float)(100 + 1);
        // if (h_probs[i] > 1 || h_probs[i] < 0) {
        //     std::cout << "warning!! probs exceed [0,1]" << std::endl;
        // }
        h_probs[i] = i;
    }
    launchTopKforBeamSearch(d_probs, batch_size, vocab_size, beamwidth, topK_workspace);
    int* res = topK_workspace + 3 * batch_size * beamwidth * 8/*max block per beam*/ * beamwidth;
    for(int i = 0; i < beamwidth; i++) {
        int id = res[i];
        printf("topK id = %d\n", id);
    }

    free(h_probs);
    cudaFree(d_probs);
    cudaFree(topK_workspace);
}