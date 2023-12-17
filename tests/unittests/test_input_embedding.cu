#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector
#include <random>

#include <cuda.h>
#include <cuda_fp16.h>
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

void cpuEmbedding(const int* input_ids, float* output, float* embed_table, const int max_context_token_num, const int hidden_size, const int vocab_size) {
    for (int i = 0; i < max_context_token_num; ++i) {
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
    const int batch_size = 1;
    const int max_context_token_num = 1024;
    const int hidden_size = 4096;
    const int vocab_size = 30000;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << " sequeue_lenght=" << max_context_token_num << "  vocab_size=" << vocab_size << std::endl;

    const int input_size = max_context_token_num;
    const int table_size = vocab_size * hidden_size;
    const int output_size = max_context_token_num * hidden_size;

    int* h_input = (int*) malloc(input_size * sizeof(int));
    // float* h_table = (float*) malloc(table_size * sizeof(float));
    // float* h_output = (float*) malloc(output_size * sizeof(float));

    half* h_table = (half*) malloc(table_size * sizeof(half));
    half* h_output = (half*) malloc(output_size * sizeof(half));

    // debug info, better to retain: 
    std::cout << "init memory on host" << std::endl;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis_int(0, vocab_size - 1);
    std::uniform_real_distribution<> dis_real(1.0, 2.0);

    for (int i = 0; i < max_context_token_num; ++i) {
        h_input[i] = dis_int(gen);
    }
    for (int i = 0; i < table_size; ++i) {
        if( i % hidden_size != 0){
            h_table[i] = (half)dis_real(gen);
            // h_table[i] = dis_real(gen);
        } else {
            h_table[i] = 1.2547e-6;
        }
    }

    int* d_input;
    // float *d_table, *d_output;
    half *d_table, *d_output;
    cudaMalloc((void**)&d_input, input_size * sizeof(int));
    // cudaMalloc((void**)&d_table, table_size * sizeof(float));
    // cudaMalloc((void**)&d_output, output_size * sizeof(float));
    cudaMalloc((void**)&d_table, table_size * sizeof(half));
    cudaMalloc((void**)&d_output, output_size * sizeof(half));
    // debug info, better to retain: 
    std::cout << "init memory on device" << std::endl;

    CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(int), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_table, h_table, table_size * sizeof(half), cudaMemcpyHostToDevice));
    // debug info, better to retain: 
    std::cout << "copy to device" << std::endl;

    DataType type_float = getTensorType<float>();
    DataType type_half = getTensorType<half>();
    DataType type_int = getTensorType<int>();
    TensorWrapper<int>* input_ids = new TensorWrapper<int>(Device::GPU, type_int, {max_context_token_num},    d_input);
    TensorWrapper<half>* output = new TensorWrapper<half>(Device::GPU, type_half, {max_context_token_num,     hidden_size}, d_output);
    EmbeddingWeight<half> emb_table;
    emb_table.data = d_table;
    launchInputEmbedding(input_ids->as<int>(), output->as<half>(), &emb_table, vocab_size);
    // debug info, better to retain:
    std::cout << "running on device" << std::endl;

    CHECK(cudaMemcpy(h_output, output->data, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    // cpuEmbedding(h_input, h_output, h_table, max_context_token_num, hidden_size, vocab_size);
    // debug info, better to retain: 
    std::cout << "printf h_output for check" << std::endl;
    std::cout << h_output[0] << std::endl;
    std::cout << h_output[1] << std::endl;
    
    // if (checkResults(h_output, (float*) output.data, output_size)) {
    //     std::cout << "Check with CPU succeed!" << std::endl;
    // } else {
    //     std::cout << "Check with CPU fail!!!" << std::endl;
    // }

    cudaFree(d_output);
    cudaFree(d_table);
    cudaFree(d_input);
    free(h_output);
    free(h_table);
    free(h_input);
    return 0;
}
