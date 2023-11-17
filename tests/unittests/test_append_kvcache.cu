#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cuda.h>
#include "src/kernels/append_to_kvcache.h"

int main() {
    const int local_batch_size = 1;
    const int max_q_len = 16;
    const int max_seq_len = 32;
    const int head_size = 8;
    const int kv_head_num = 2;
    const int kv_size = 1 * local_batch_size * max_q_len * kv_head_num * head_size;
    const int layer_offset = 1 * local_batch_size * max_seq_len * kv_head_num * head_size;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    const int kvcache_size = layer_offset;
    const int layer_id = 0;

    float* h_k_src;
    float *d_k_src;
    h_k_src = (float*)malloc(sizeof(float) * kv_size);
    cudaMalloc((void**)&d_k_src, sizeof(float) * kv_size);

    float* h_v_src;
    float *d_v_src;
    h_v_src = (float*)malloc(sizeof(float) * kv_size);
    cudaMalloc((void**)&d_v_src, sizeof(float) * kv_size);

    int* cur_query_length = (int*)malloc(sizeof(int) * local_batch_size);
    int* history_length = (int*)malloc(sizeof(int) * local_batch_size);
    int* dcur_query_length;
    int* dhistory_length;
    cudaMalloc((void**)&dcur_query_length, sizeof(int) * local_batch_size);
    cudaMalloc((void**)&dhistory_length, sizeof(int) * local_batch_size);
   
    float* h_k_dst = (float*)malloc(sizeof(float) * kvcache_size);
    float* h_v_dst = (float*)malloc(sizeof(float) * kvcache_size);
    float* d_k_dst;
    float* d_v_dst;
    cudaMalloc((void**)&d_k_dst,sizeof(float) * kvcache_size);
    cudaMalloc((void**)&d_v_dst,sizeof(float) * kvcache_size);
    float* kv_scale;
    cudaMalloc((void**)&kv_scale, sizeof(float));
    for(int i = 0; i < kv_size; i++) {
       h_k_src[i] = 1.0f;
       h_v_src[i] = 1.0f;
    }
    for(int i = 0; i < local_batch_size; i++) {
       cur_query_length[i] = 16;
       history_length[i] = 1;
    }
    int* h_layer_id = (int*)malloc(sizeof(int)*local_batch_size);
    int* d_layer_id;
    cudaMalloc((void**)&d_layer_id,sizeof(int)*local_batch_size);

    cudaMemcpy(d_v_src, h_v_src, sizeof(float)*kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_src, h_k_src, sizeof(float)*kv_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dcur_query_length, cur_query_length, sizeof(int)*local_batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dhistory_length, history_length, sizeof(int)*local_batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_layer_id, h_layer_id, sizeof(int)*local_batch_size, cudaMemcpyHostToDevice);

    DataType type = getTensorType<float>(); 
    DataType type_int = getTensorType<int>(); 
    Tensor in_ksrc(Device::GPU, type, {local_batch_size, kv_head_num, max_q_len, head_size}, d_k_src);
    Tensor in_vsrc(Device::GPU, type, {local_batch_size, kv_head_num, max_q_len, head_size}, d_v_src);
    Tensor layer_id(Device::GPU, type_int, {local_batch_size}, d_layer_id);
    Tensor cur_q_len(Device::GPU, type_int, {local_batch_size}, dcur_query_length);
    Tensor history_len(Device::GPU, type_int, {local_batch_size}, dhistory_length);
    Tensor out_kdst(Device::GPU, type, {local_batch_size, kv_head_num, max_seq_len, head_size}, d_k_dst);
    Tensor out_vdst(Device::GPU, type, {local_batch_size, kv_head_num, max_seq_len, head_size}, d_v_dst);
    // size_t layer_offset = layer_id * local_batch_size * kv_head_num * max_seq_len * head_size;
    // debug info, better to retain: std::cout << "before launch kernel" << std::endl;
    launchAppendKVCache(&in_ksrc, &in_vsrc, &layer_id, &cur_q_len, &history_len, &out_kdst, &out_vdst);
    // debug info, better to retain: std::cout << "after launch kernel" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_v_dst, d_v_dst, sizeof(float) * kvcache_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_k_dst, d_k_dst, sizeof(float) * kvcache_size, cudaMemcpyDeviceToHost);
    // debug info, better to retain: std::cout << "cuda memcpy device to host" << std::endl;
    // note: need to add offset2index and index2offset API to help us program and check result
    for(int i = local_batch_size * (1) * kv_head_num * head_size; i < local_batch_size * max_seq_len * kv_head_num * head_size; i++) {
        printf("index = %d\n", i);
        printf("res k = %f\n",h_k_dst[i]);
        // debug info, better to retain: printf("topK id = %d\n", id);
        printf("res v = %f\n",h_v_dst[i]);
        printf("===============\n");
        // debug info, better to retain: printf("topK val =%f\n", val);
    }
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_k_src);
    free(h_v_src);
    free(h_k_dst);
    free(h_v_dst);
    free(cur_query_length);
    free(history_length);
    free(h_layer_id);
    cudaFree(d_k_src);
    cudaFree(d_v_src);
    cudaFree(d_k_dst);
    cudaFree(d_v_dst);
    cudaFree(dcur_query_length);
    cudaFree(dhistory_length);
    cudaFree(kv_scale);
    cudaFree(d_layer_id);
}
