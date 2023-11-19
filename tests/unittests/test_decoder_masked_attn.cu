#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include "src/kernels/decoder_masked_attn.h"
// bug1: MUST add CHECK to cudaMemcpy to see if its work well
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

void CPUMaskedAttn(const float* q,
                    const float* k,
                    const float* v,
                    float* k_cache,
                    float* v_cache,
                    float* mha_output,
                    const int batch_size,
                    const int num_heads,
                    const int head_size,
                    const int step){
    int batch_stride = num_heads * head_size;
    int head_stride = head_size;
    int cache_offset = batch_size * batch_stride;
    int block_nums = batch_size * num_heads;
    float scale = rsqrt(float(head_size));

    const float* q_mem = q;
    const float* k_mem = k;
    const float* v_mem = v;

    // tmp buffer
    float* sqk = (float*)malloc(sizeof(float) * (block_nums * (3 * head_size + step)));
    float* sq = sqk;
    float* sk = sq + block_nums * head_size;
    float* logits = sk + block_nums * head_size;
    float* sv = logits + block_nums * step;
    // FT 2.1的写法里面，kv cache是在prompt阶段已经填充，iter=0为token gen的起始iter
    for(int batch_id = 0; batch_id < batch_size; batch_id++) {
        for(int head_id = 0; head_id < num_heads; head_id++) {
            float row_max = 0.0f;
            for(int iter = 0; iter < step; iter++) {
                float attn_score = 0.0f;
                for(int tid = 0; tid < head_size; tid++) {
                    int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
                    // note: sq and sk's offset should be qkv_offset , not tid
                    sk[qkv_offset]= k_cache[iter * cache_offset + qkv_offset];
                    // when final step, update k cache
                    if (iter == step - 1) {
                        // TODO: update k cache with k with bias add
                        k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
                        sk[qkv_offset] = k_mem[qkv_offset];
                    }
                    
                    sq[qkv_offset] = q_mem[qkv_offset];
                    float qk = sq[qkv_offset] * sk[qkv_offset] * scale;
                    //block reduce using multi warp reduce
                    //TODO: maybe broadcast the attn score to each thread of the block in blockreducesum
                    attn_score += qk;
                }
                // note: logtis's offset should be as follow, not should mul head size with iter
                //debug info,printf("every step/seqlen attn score = %f\n", attn_score);
                logits[batch_id * num_heads * step + head_id * step + iter] = attn_score;
                //softmax(logits), logits.shape = [bs, num heads, 1, step] 
                row_max = std::max(attn_score, row_max);
            }
            printf("all step/seqlen(one row) max attn score = %f\n", row_max);
            float fenzi = 0.0f;
            float fenmu = 0.0f;
            for(int iter = 0; iter < step; iter++) { // row
                fenzi = expf(logits[batch_id * num_heads * step + head_id * step + iter] - row_max);
                fenmu += fenzi;
            }
            for(int iter = 0; iter < step; iter++) { // row
                logits[batch_id * num_heads * step + head_id * step + iter] = fenzi / fenmu;
                printf("logits=%f\n",fenzi/fenmu);
            }
            // logits*V = [bs, num heads, 1, step] * [mx_seq_len or step, bs, num heads, head size]
            //for(int iter = 0; iter < step; iter++) { 
            for(int tid = 0; tid < head_size; tid++) {
                float O = 0.0f;
                int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
                for(int iter = 0; iter < step; iter++) {
                    sv[qkv_offset]= v_cache[iter * cache_offset + qkv_offset];
                    // when final step, update k cache
                    if (iter == step - 1) {
                    // TODO: update k cache with k with bias add
                        v_cache[iter * cache_offset + qkv_offset] = v_mem[qkv_offset];
                        sv[qkv_offset] = v_mem[qkv_offset];
                    }
                    O += sv[qkv_offset] * logits[batch_id * num_heads * step + head_id * step + iter];
                    printf("logits[%d]=%f, sv[%d]=%f, O=%f\n",iter,logits[iter],qkv_offset,sv[qkv_offset],O);
                }
                mha_output[qkv_offset] = O;
            }
        }
    }

    free(sqk);
}

bool CheckResult(float* CPUoutput, float* GPUoutput, int output_size) {
    for(int i = 0; i < output_size; i++) {
        if(fabs(CPUoutput[i] - GPUoutput[i]) > 1e-6){    
            printf("the %dth res is wrong, CPUoutput = %f, GPUoutput = %f\n", i, CPUoutput[i], GPUoutput[i]);
            return false;
        }
    }
    return true;
}
int main() {
    constexpr int batch_size = 1;
    constexpr int head_size = 16;
    constexpr int num_heads = 2;
    constexpr int step = 4;
    constexpr int max_seq_len = 32;
    float* h_q;
    float* d_q;
    int q_size = batch_size * num_heads * head_size;
    h_q = (float*)malloc(sizeof(float) * q_size);
    cudaMalloc((void**)&d_q, sizeof(float) * q_size);

    float* h_k;
    float* d_k;
    int k_size = batch_size * num_heads * head_size;
    h_k = (float*)malloc(sizeof(float) * k_size);
    cudaMalloc((void**)&d_k, sizeof(float) * k_size);

    float* h_v;
    float* d_v;
    int v_size = batch_size * num_heads * head_size;
    h_v = (float*)malloc(sizeof(float) * v_size);
    cudaMalloc((void**)&d_v, sizeof(float) * v_size);  

    float* h_kcache;
    float* d_kcache;
    int kcache_size = max_seq_len * batch_size * num_heads * head_size;
    h_kcache = (float*)malloc(sizeof(float) * kcache_size);
    cudaMalloc((void**)&d_kcache, sizeof(float) * kcache_size);  

    float* h_vcache;
    float* d_vcache;
    int vcache_size = max_seq_len * batch_size * num_heads * head_size;
    h_vcache = (float*)malloc(sizeof(float) * vcache_size);
    cudaMalloc((void**)&d_vcache, sizeof(float) * vcache_size);  

    for(int i = 0; i < q_size; i++) { // initialize host data
        h_q[i] = 1.0f;
        h_k[i] = 1.0f;
        h_v[i] = 1.0f;
    }
    // note: prompt phase only generate part of k v cache
    for(int i = 0; i < (kcache_size * step) / max_seq_len; i++) { // initialize host data
        h_kcache[i] = 1.0f;
        h_vcache[i] = 1.0f;
    }

    float* h_o;
    float* d_o;
    int o_size = batch_size * num_heads * head_size;
    h_o = (float*)malloc(sizeof(float) * o_size);
    cudaMalloc((void**)&d_o, sizeof(float) * o_size); 

    cudaMemcpy(d_q, h_q, sizeof(float) * q_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, h_k, sizeof(float) * k_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, h_v, sizeof(float) * v_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kcache, h_kcache, sizeof(float) * kcache_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vcache, h_vcache, sizeof(float) * vcache_size, cudaMemcpyHostToDevice);
  
    launchDecoderMaskedMHA(d_q, d_k, d_v, d_kcache, d_vcache, d_o, batch_size, num_heads, head_size, step);
    CHECK(cudaMemcpy(h_o, d_o, sizeof(float) * o_size, cudaMemcpyDeviceToHost));
    float* CPU_output = (float*)malloc(sizeof(float) * o_size);
    CPUMaskedAttn(h_q, h_k, h_v, h_kcache, h_vcache, CPU_output, batch_size, num_heads, head_size, step);
    bool is_true = CheckResult(CPU_output, h_o, o_size);
    if(is_true){
        printf("test passed");
    } else {
        printf("test failed");
    }

    free(h_q);
    free(h_k);
    free(h_v);
    free(h_kcache);
    free(h_vcache);
    free(h_o);
    free(CPU_output);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_o);
    cudaFree(d_kcache);
    cudaFree(d_vcache);
}
