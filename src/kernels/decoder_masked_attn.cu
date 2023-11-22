#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/kernels/decoder_masked_attn.h"
// bug1: scale's dtype must be float ,not int
// bug2: mha_kernel_params struct's pointer is on CPU, not GPU, which cause we dont run the cuda kernel, so add cudacheck is a must!
// bug3: blockreduce res should use tid=0 to write into smem
template<typename T>
__device__ T warpReduceSum(T val){

    for(int mask = 16; mask > 0; mask >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;

}
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpsum[64];//why add static?or will report incomplete type
    // returned val is the sum computed by 0th thread.
    val = warpReduceSum<T>(val);
    //note: here return val of warpreducesum should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpsum[warp_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warpsum[tid] : 0;
    return warpReduceSum<T>(warp_val);

}
template<typename T>
__device__ T warpReduceMax(T val){

    for(int mask = 16; mask > 0; mask >>= 1){
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T>
__device__ T blockReduceMax(T val){
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = (blockDim.x + 31)/ 32;
    static __shared__ T warpmax[64];
    // returned val is the max computed by 0th thread.
    val = warpReduceMax<T>(val);
    //note: here return val of warpreducemax should be stored into smem , rather not reg, because here nums of return val  are warp nums not thread nums.
    if (lane_id == 0){
        warpmax[warp_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warpmax[tid] : 0;
    return warpReduceMax<T>(warp_val);
}
// kv cache is the output of context attention(prompt phase), and the input of masked attention(token gen)
// struct masked_MHA_kernel_params
// {
//     float* q;       //[bs, num heads, 1, head size]
//     float* k;       //[bs, num heads, 1, head size]
//     float* v;       //[bs, num heads, 1, head size]
//     float* k_cache; //output,[max_seq_len or step, bs, num heads, head size] from prompt phase
//     float* v_cache; //output,[max_seq_len or step, bs, num heads, head size] from prompt phase
//     int batch_size;
//     int num_heads;
//     int head_size;
//     float scale; // =rsqrt(head size);
//     //TODO: add qkv bias
//     int step;
//     float* mha_output;
//};
// block and thread allocation
// 1 block -> head size，后续可改进为1 warp -> 1 head size
// 1 grid -> bs * num heads
__global__ void masked_MHA_kernel(const float* q,
                    const float* k,
                    const float* v,
                    float* k_cache,
                    float* v_cache,
                    float* mha_output,
                    const int batch_size,
                    const int num_heads,
                    const int head_size,
                    const int step,
                    float scale){
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int head_id = bid % num_heads;
    int batch_id = bid / num_heads;

    int batch_stride = num_heads * head_size;
    int head_stride = head_size;
    int qkv_offset = batch_id * batch_stride + head_id * head_stride + tid;
    int cache_offset = batch_size * batch_stride;

    const float* q_mem = q;
    const float* k_mem = k;
    const float* v_mem = v;
    // q k smem for block reduce
    extern __shared__ float sqk[];
    float* sq = sqk;
    float* sk = sq + head_size;
    float* logits = sk + head_size;
    float* sv = logits + step;
    // FT 2.1的写法里面，kv cache是在prompt阶段已经填充，iter=0为token gen的起始iter
    for(int iter = 0; iter < step; iter++) {
        // every iter,  q and k's shape = [1, head size]
        // reuse k cache
        // float k = k_cache[iter * cache_offset + qkv_offset];
        //或许可以在每个step省略掉前step-1的qk dot
        //if(tid < head_size){
        sk[tid]= k_cache[iter * cache_offset + qkv_offset];
        __syncthreads();
        // when final step, update k cache
        if (iter == step - 1) {
            // TODO: update k cache with k with bias add
            k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
            sk[tid] = k_mem[qkv_offset];
        }

        sq[tid] = q_mem[qkv_offset];
        __syncthreads();
        float qk = (tid < head_size) ? sq[tid] * sk[tid] * scale : (float)0.0f;
        //block reduce using multi warp reduce
        //TODO: maybe broadcast the attn score to each thread of the block in blockreducesum
        float attn_score = blockReduceSum<float>(qk);
        if(tid == 0) {
            logits[iter] = attn_score;
        }
        __syncthreads();
    }
    //softmax(logits), logits.shape = [bs, num heads, 1, step]
    float local_logits = tid < step ? (float)logits[tid] : 0;
    __shared__ float row_max, fenmu;
    
    float block_max = blockReduceMax<float>(local_logits);
    if (tid == 0){
        row_max = block_max;
    }
    float fenzi = tid < step ? expf(logits[tid] - row_max) : 0;
    
    float block_fenmu = blockReduceSum<float>(fenzi);
    if (tid == 0){
        fenmu = block_fenmu;
    }
    if(tid < step) {
        logits[tid] = (float)(fenzi / fenmu);
    }
    __syncthreads();

    // logits*V = [bs, num heads, 1, step] * [max_seq_len or step, bs, num heads, head size]
    if (tid < head_size) {
        // note: here is head size ,not step, because step by step, we have to use [1, step/seqlen] from logits * [1, head size] from v
        // so here we use acc O to acc the one ele logits * one ele v every step iter
        float O = 0.0f;
        for(int iter = 0; iter < step; iter++) {
            sv[tid]= v_cache[iter * cache_offset + qkv_offset];
            __syncthreads();
            // when final step, update k cache
            if (iter == step - 1) {
                // TODO: update k cache with k with bias add
                v_cache[iter * cache_offset + qkv_offset] = v_mem[qkv_offset];
                sv[tid] = v_mem[qkv_offset];
            }
            //if(bid==0 && tid == 0){
            //printf("when tid=0, v cache = %f\n", sv[tid]);

            O += sv[tid] * logits[iter];
            __syncthreads();
        }
        mha_output[qkv_offset] = O;
    }
}

void launchDecoderMaskedMHA(float* q,
                            float* k,
                            float* v,
                            float* k_cache,
                            float* v_cache,
                            float* mha_output,
                            const int batch_size,
                            const int num_heads,
                            const int head_size,
                            const int step){
    float scale = rsqrt(float(head_size));

    dim3 grid(batch_size * num_heads);
    dim3 block(head_size);
    printf("calling fused masked self attn kernel\n");
    masked_MHA_kernel<<<grid, block, (3 * head_size + step) * sizeof(float)>>>(q,
                                                                                k,
                                                                                v,
                                                                                k_cache,
                                                                                v_cache,
                                                                                mha_output,
                                                                                batch_size,
                                                                                num_heads,
                                                                                head_size,
                                                                                step,
                                                                                scale);
    printf("called fused masked self attn kernel\n");
}

