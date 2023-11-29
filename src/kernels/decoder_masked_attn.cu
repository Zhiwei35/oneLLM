#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/kernels/decoder_masked_attn.h"
// bug1: scale's dtype must be float ,not int
// bug2: mha_kernel_params struct's pointer is on CPU, not GPU, which cause we dont run the cuda kernel, so add cudacheck is a must!
// bug3: blockreduce res should use tid=0 to write into smem
// bug4: GQA, kv_head_num brd to head_num, we can automaticly do this by head id index like lmdeploy
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
//     float* q;       //[bs, q num heads, 1, head size]
//     float* k;       //[bs, kv num heads, 1, head size]
//     float* v;       //[bs, num heads, 1, head size]
//     float* k_cache; //output,[max_seq_len or step, bs, kv num heads, head size] from prompt phase
//     float* v_cache; //output,[max_seq_len or step, bs, num heads, head size] from prompt phase
//     int batch_size;
//     int num_heads;
//     int head_size;
//     float scale; // =rsqrt(head size);
//     //TODO: add qkv bias
//     int step;
//     float* mha_output;
//};
template<typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0;
};

template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};

struct TwoFloat2{
    float2 x;
    float2 y;
};

inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

// RoPE公式决定必须要做向量化
inline __device__ float2 GetRoPEres(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}


inline __device__ void apply_RoPE(float4& q, float4& k, int tid, int rot_embed_dim, float base, float t_step){
    if(4 * tid >= rot_embed_dim){
        return;
    }


    TwoFloat2& q_ = *reinterpret_cast<TwoFloat2*>(&q); // q为float4 寄存器
    TwoFloat2& k_ = *reinterpret_cast<TwoFloat2*>(&k);
    
    float2 coef0 = GetRoPEfreq(4 * tid, rot_embed_dim, base, t_step);
    // float freq0 = timestep / powf(rotary_embedding_base, 4 * tid / (float) rotary_embedding_dim); //分子zid = 0,2,4,, headsize/2-1,对应的theta下标为0,1,2.对应的headsize维度的索引为(0,1),(2,3)
    q_.x = GetRoPEres(q_.x ,coef0);
    // rot0.x = coef0.x * q.x -  coef0.y * q.y; //q.x为x0,q.y为x1，head size维度上两个相邻
    // rot0.y = coef0.x * q.y +  coef0.y * q.x
    float2 coef1 = GetRoPEfreq(4 * tid + 2, rot_embed_dim, base, t_step);
    q_.y = GetRoPEres(q_.y ,coef1);
    // rot1.x = coef1.x * q.x -  coef1.y * q.y; //q.x为x2,q.y为x3，head size维度上两个相邻
    // rot1.y = coef1.x * q.y +  coef1.y * q.x;
    k_.x = GetRoPEres(k_.x ,coef0);
    k_.y = GetRoPEres(k_.y ,coef1);
}

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
                    const int head_num,
                    const int kv_head_num,
                    //const int num_heads,
                    const int head_size,
                    const int step,
                    float scale){// rsqrt(dh)
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int q_head_id = bid % head_num;
    int q_batch_id = bid / head_num;
    int kv_head_id = bid % kv_head_num;
    int kv_batch_id = bid / kv_head_num;

    int batch_stride = num_heads * head_size;
    int head_stride = head_size;
    int q_offset = q_batch_id * batch_stride + q_head_id * head_stride + tid;
    int k_offset = kv_batch_id * batch_stride + kv_head_id * head_stride + tid;
    int cache_offset = batch_size * batch_stride;

    int vec_size = Vec<float>::size;
    int q_offset_vec = q_batch_id * batch_stride + q_head_id * head_stride + tid * vec_size;
    int k_offset_vec = k_batch_id * batch_stride + kv_head_id * head_stride + tid * vec_size;

    using Vec_t = typename Vec<float>::Type;
    Vec_t qvec, kvec;
    //reuse q k v reg from rope
    const float* q_mem = q;
    const float* k_mem = k;
    const float* v_mem = v;
    if (tid * vec_size < head_size) {
        qvec = *reinterpret_cast<Vec_t*>(&q_mem[q_offset_vec]);
        kvec = *reinterpret_cast<Vec_t*>(&k_mem[k_offset_vec]);
        apply_RoPE(qvec, kvec, tid, rotary_embedding_dim, rotary_embedding_base, timestep);
    }
    // q k smem for block reduce
    extern __shared__ float sqk[];
    float* sq = sqk;
    float* sk = sq + head_size;
    float* logits = sk + head_size;
    float* sv = logits + step;
    //sq[tid] = q_mem[qkv_offset];
    if (tid * vec_size < head_size) {
        *reinterpret_cast<Vec_t*>(&sq[tid * vec_size]) = qvec;
    }
    __syncthreads();
    // FT 2.1的写法里面，kv cache是在prompt阶段已经填充，iter=0为token gen的起始iter
    for(int iter = 0; iter < step; iter++) {
        // every iter,  q and k's shape = [1, head size]
        // reuse k cache
        // float k = k_cache[iter * cache_offset + qkv_offset];
        //或许可以在每个step省略掉前step-1的qk dot
        sk[tid]= k_cache[iter * cache_offset + k_offset];
        __syncthreads();
        // when final step, update k cache
        if (iter == step - 1 && tid * vec_size < head_size) {
            // TODO: update k cache with k with bias add
            //k_cache[iter * cache_offset + qkv_offset] = k_mem[qkv_offset];
            //sk[tid] = k_mem[qkv_offset];
            *reinterpret_cast<Vec_t*>(&k_cache[iter * cache_offset + k_offset_vec]) = kvec;
            *reinterpret_cast<Vec_t*>(&sk[tid * vec_size]) = kvec;         
        }

        // sq[tid] = q_mem[qkv_offset];
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
            sv[tid]= v_cache[iter * cache_offset + k_offset];
            __syncthreads();
            // when final step, update k cache
            if (iter == step - 1) {
                // TODO: update k cache with k with bias add
                v_cache[iter * cache_offset + k_offset] = v_mem[k_offset];
                sv[tid] = v_mem[k_offset];
            }
            //if(bid==0 && tid == 0){
            //printf("when tid=0, v cache = %f\n", sv[tid]);

            O += sv[tid] * logits[iter];
            __syncthreads();
        }
        mha_output[q_offset] = O;
    }
}

// void launchDecoderMaskedMHA(float* q,
//                             float* k,
//                             float* v,
//                             float* k_cache,
//                             float* v_cache,
//                             float* mha_output,
//                             const int batch_size,
//                             const int num_heads,
//                             const int head_size,
//                             const int step){
void launchDecoderMaskedMHA(Tensor* qkv_buf,
                            Tensor* k_cache,
                            Tensor* v_cache,
                            Tensor* finished,
                            Tensor* step,
                            Tensor* mha_output){
    const int batch_size = qkv_buf->shape[0];
    const int qkv_head_num = qkv_buf->shape[1];
    const int kv_head_num = k_cache->shape[2]; 
    int head_num = qkv_head_num - 2 * kv_head_num;
    const int head_size = qkv_buf->shape[2];
    const int cur_step = *(int*)step->data;
    float* qkv_data = (float*)qkv_buf->data;
    //[bs,1,qkv_head_num,head_size]
    float* q = qkv_data;
    float* k = qkv_data + head_num * head_size;
    float* v = qkv_data + (head_num + kv_head_num) * head_size;

    float scale = rsqrt(float(head_size));

    dim3 grid(batch_size * head_num);//这里的block分配可以匹配得上lmdeploy
    dim3 block(head_size); //vec size = 4 for fp32
    printf("calling fused masked self attn kernel\n");
    masked_MHA_kernel<<<grid, block, (3 * head_size + step) * sizeof(float)>>>(q,
                                                                                k,
                                                                                v,
                                                                                (float*)k_cache->data,
                                                                                (float*)v_cache->data,
                                                                                (float*)mha_output->data,
                                                                                batch_size,
                                                                                head_num,
                                                                                kv_head_num,
                                                                                //num_heads,
                                                                                head_size,
                                                                                cur_step,
                                                                                scale);
    printf("called fused masked self attn kernel\n");
}

