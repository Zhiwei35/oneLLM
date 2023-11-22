#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>

#include "src/kernels/input_embedding.h"

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

void cpuEmbedding(const int* input_ids, Tensor* output, Tensor* embed_table, const int sequeue_length, const int hidden_size, const int vocab_size) {
    for (int i = 0; i < sequeue_length; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            output[j + i * sequeue_length * hidden_size] = embed_table[j + input_ids[i] * hidden_size];
        }
    }
}

bool checkResults(Tensor* r1, Tensor* r2, const int length) {
    for (int i = 0; i < length; ++i) {
        if (r1[i] != r2[i]) return false;
    }
    return true;
}

int main() {
    const int sequeue_length = 1024;
    const int hidden_size = 4096;
    const int vocab_size = 30000;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << " sequeue_lenght=" << sequeue_length << "  vocab_size=" << vocab_size << std::endl;

    const int input_size = sequeue_length;
    const int table_size = vocab_size * hidden_size;
    const int output_size = sequeue_length * hidden_size;

    int* h_input = (int*) malloc(input_size * sizeof(int));
    float* h_table = (float*) malloc(table_size * sizeof(float));
    float* h_output = (float*) malloc(output_size * sizeof(float));
    float* output = (float*) malloc(output_size * sizeof(float)); // prepare for cpu check
    // debug info, better to retain: 
    std::cout << "init memory on host" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_int(0, vocab_size - 1);
    std::uniform_real_distribution<> dis_real(1.0, 2.0);

    for (int i = 0; i < sequeue_length; ++i) {
        h_input[i] = dis_int(gen);
    }
    for (int i = 0; i < table_size; ++i) {
        h_table[i] = dis_real(gen);
    }

    int* d_input;
    float* d_table, d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(int));
    cudaMalloc((void**)&d_table, table_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));
    // debug info, better to retain: 
    std::cout << "init memory on device" << std::endl;

    CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice));

    launchInputEmbedding(d_input, d_output, d_table, sequeue_length, hidden_size, vocab_size);
    // debug info, better to retain: 
    std::cout << "running the device" << std::endl;

    cpuEmbedding(h_input, output, h_table, sequeue_length, hidden_size, vocab_size);

    CHECK(cudaMemcpy(h_output, h_output, output_size * sizeof(output), cudaMemcpyDeviceToHost));
    
    std::cout << checkResults(h_output, output, output_size) ? "Check with CPU succeed!" : "CHeck with CPU fail!!!" << std::endl;

    cudaFree(d_output);
    cudaFree(d_table);
    cudaFree(d_input);
    free(output);
    free(h_output);
    free(h_table);
    free(h_input);
    return 0;
}
