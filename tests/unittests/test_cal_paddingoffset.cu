#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include "src/kernels/cal_paddingoffset.h"

int main() {
    const int batch_size = 3;
    const int max_q_len = 5;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << "  vocab_size=" << vocab_size << std::endl;
    int* h_seq_lens;
    int *d_seq_lens;
    h_seq_lens = (int*)malloc(sizeof(int) * batch_size);
    cudaMalloc((void**)&d_seq_lens, sizeof(int) * batch_size);

    int* h_cum_seqlens;
    int* d_cum_seqlens;
    h_cum_seqlens = (int*)malloc(sizeof(int) * (batch_size + 1));
    cudaMalloc((void**)&d_cum_seqlens, sizeof(int) * (batch_size + 1));
    
    int* h_padding_offset;
    int* d_padding_offset;
    h_padding_offset = (int*)malloc(sizeof(int) * batch_size * max_q_len);
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * batch_size * max_q_len);

    int* h_pinned_token_num;
    cudaMallocHost((void**)&h_pinned_token_num, sizeof(int));

    for(int i = 0; i < batch_size; i++) { // 3
       h_seq_lens[i] = batch_size;
    }
    cudaMemcpy(d_seq_lens, h_seq_lens, sizeof(int) * batch_size, cudaMemcpyHostToDevice);
    // debug info, better to retain: std::cout << "before launch kernel" << std::endl;
    int h_token_num{};
    launchCalPaddingoffset(h_pinned_token_num, 
                           &h_token_num, 
                           d_padding_offset, 
                           d_cum_seqlens,
                           d_seq_lens, 
                           batch_size,
                           max_q_len);
    // debug info, better to retain: std::cout << "after launch kernel" << std::endl;
    // Note: remember to memcpy from device to host and define the correct copy size(mul the sizeof(dtype)), or will cause segment fault
    cudaMemcpy(h_padding_offset, d_padding_offset, sizeof(int) * batch_size * max_q_len, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_cum_seqlens, d_cum_seqlens, sizeof(int) * (batch_size + 1), cudaMemcpyDeviceToHost);
    // debug info, better to retain: std::cout << "cuda memcpy device to host" << std::endl;    
    for(int i = 0; i < batch_size * max_q_len; i++) {
        printf("padding_offset = %d\n", h_padding_offset[i]);
    }
    for(int i = 0; i < batch_size + 1; i++){
        printf("cum_seqlens =%d\n", h_cum_seqlens[i]);
    }
    //expected result is:
    // padding_offset: 0,0,0,2,2,2,4,4,4,0.... shape = [batchsize, max_q_len]
    // cum_seqlens: 0,3,6,9. shape=[batchsize+1]
    // debug info, better to retain: std::cout << "before free" << std::endl;
    free(h_seq_lens);
    free(h_padding_offset);
    free(h_cum_seqlens);
    cudaFreeHost(h_pinned_token_num);
    cudaFree(d_seq_lens);
    cudaFree(d_padding_offset);
    cudaFree(d_cum_seqlens);
}
