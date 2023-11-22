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

void cpuEmbedding(const int* input_ids, float* output, float* embed_table, const int sequeue_length, const int hidden_size, const int vocab_size) {
    for (int i = 0; i < sequeue_length; ++i) {
        for (int j = 0; j < hidden_size; ++j) {
            output[j + i * hidden_size] = embed_table[j + input_ids[i] * hidden_size];
        }
    }
}

bool checkResults(float* h_output, float* d_output, const int output_size) {
    float* d_output_cpu = (float*) malloc(output_size * sizeof(float)); // prepare for cpu check
    CHECK(cudaMemcpy(d_output_cpu, d_output, output_size * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < output_size; ++i) {
        if (fabs(d_output_cpu[i] - h_output[i]) > 1e5) {
            std::cout << "Dev : ";
            for (int j = max(0, i - 10); j < min(output_size, i + 10); ++j) {
                std::cout << d_output_cpu[i];
            }
            std::cout << std::endl;
            std::cout << "Cpu : ";
            for (int j = max(0, i - 10); j < min(output_size, i + 10); ++j) {
                std::cout << h_output[i];
            }
            std::cout << std::endl;
            free(d_output_cpu);
            return false;
        }
    }
    free(d_output_cpu);
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
    float *d_table, *d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(int));
    cudaMalloc((void**)&d_table, table_size * sizeof(float));
    cudaMalloc((void**)&d_output, output_size * sizeof(float));
    // debug info, better to retain: 
    std::cout << "init memory on device" << std::endl;

    CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice));
    // debug info, better to retain: 
    std::cout << "copy to device" << std::endl;

    launchInputEmbedding(d_input, d_output, d_table, sequeue_length, hidden_size, vocab_size);
    // debug info, better to retain: 
    std::cout << "running on device" << std::endl;

    cpuEmbedding(h_input, h_output, h_table, sequeue_length, hidden_size, vocab_size);
    // debug info, better to retain: 
    std::cout << "running cpu for check" << std::endl;
    
    if (checkResults(h_output, d_output, output_size)) {
        std::cout << "Check with CPU succeed!" << std::endl;
    } else {
        std::cout << "Check with CPU fail!!!" << std::endl;
    }

    cudaFree(d_output);
    cudaFree(d_table);
    cudaFree(d_input);
    free(h_output);
    free(h_table);
    free(h_input);
    return 0;
}
