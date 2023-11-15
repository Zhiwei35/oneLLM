#include "src/kernels/attn_softmax_kernel.h"
#include <float.h>
#include <math.h>
#include <assert.h>
#include <stdio.h>
// attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
// qk,                 (batch_size, head_num, q_length, k_length), QK^T.
// attention_mask,     (batch_size, q_length, k_length), attention mask.
template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> class ReductionOp, typename T>
__inline__ __device__ T warpReduce(T val) {
  for (int mask = 32 / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<template<typename> class ReductionOp, typename T>
__inline__ __device__ T blockReduce(T val) {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int warp_nums = blockDim.x / 32;
    static __shared__ T warp[64];
    val = warpReduce<ReductionOp, T>(val);
    //note: here return val of warpreducesum should be stored into smem , rather not reg, because here nums of return val are warp nums not thread nums.
    if (lane_id == 0){
        warp[warp_id] = val;
    }
    __syncthreads();
    float warp_val = tid < warp_nums ? warp[warp_id] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}


template<int NUMS_PER_THREAD_PER_ROW>
__global__ void ScaleMaskAndSoftmax(float* attn_score,
                                    float* qk,
                                    uint8_t* mask,
                                    int batch_size,
                                    int head_nums,
                                    int q_len,
                                    int k_len,
                                    float scale)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    //note: NUMS_PER_THREAD_PER_ROW must be a constant value that known at compile time, following expr is invalid
    //const int NUMS_PER_THREAD_PER_ROW = ceil(k_len / blockDim.x);
    __shared__ float inv_sum, s_max;
    //warning: remember 1st priority thing is filtering the out-of-boundary threads
    if(threadIdx.x >= k_len){
        return;
    }
    for(int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) {
        int qk_offset = 0;
        int mask_offset = 0;
        float qk_data = static_cast<float>(0);
        uint8_t mask_data = static_cast<uint8_t>(0);
        float thread_max = FLT_MIN;
        float data[NUMS_PER_THREAD_PER_ROW]; // 面对这种一个block一个thread需要处理多行多列的时候，数据尽量用数组存储，计算出每个block和thread要处理几行几列
        //for(int col_start = threadIdx.x; col_start < k_len; col_start += blockDim.x){
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){   
            qk_offset = batch_id * head_nums * q_len * k_len + head_id * q_len * k_len
                            + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            qk_data = qk[qk_offset];
            
            mask_offset = batch_id * q_len * k_len + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            mask_data = mask[mask_offset];
            
            //debug info,printf("before,data[%d]=%f\n",col_start, data[col_start]);
            data[col_start] = scale * qk_data + (float)mask_data;
            //debug info,printf("after,scale*qk_data=%f, (float)mask_data=%f,data[%d]=%f\n",scale * qk_data, (float)mask_data, col_start, data[col_start]);
            thread_max = fmax(data[col_start], thread_max);
        }
        // warp/block reduce
        float max_val = blockReduce<MaxOp, float>(thread_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
            //debug info,printf("row max = %f\n", s_max);
        }
        __syncthreads();
        // thread local fenzi/fenmu
        float thread_sum = 0.0f;
        //for(int col_start = threadIdx.x; col_start < k_len; col_start += blockDim.x){
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){ 
            //debug info to see useless threads if its available,printf("blockIdx.x=%d, threadIdx.x=%d\n",blockIdx.x, threadIdx.x);
            
            qk_offset = batch_id * head_nums * q_len * k_len + head_id * q_len * k_len
                            + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            mask_offset = batch_id * q_len * k_len + row_start * k_len + col_start * blockDim.x +         threadIdx.x;
            data[col_start] = expf(data[col_start] - s_max);
            thread_sum += data[col_start];
            //debug info,printf("after, data[%d]=%f, thread_sum = %f\n",col_start, data[col_start], thread_sum);
        }
        // row sum
        float sum = blockReduce<SumOp, float>(thread_sum);
        if(threadIdx.x == 0) {
            inv_sum = 1 / sum;//maybe sum(fenmu) need to add a small value to keep stable
            //debug info, printf("row sum = %f\n", sum);
        }
        __syncthreads();
        // write back into gmem
       for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){ 
            qk_offset = batch_id * head_nums * q_len * k_len + head_id * q_len * k_len
                            + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            attn_score[qk_offset] = (data[col_start] * inv_sum);
        }
    }
}

void launchScaleMaskAndSoftmax(Tensor* qk,
                               Tensor* mask,
                               Tensor* attn_score,
                               float scale)
{
    // attention_score,    (batch_size, head_num, q_length, k_length), softmax output.
    // qk,                 (batch_size, head_num, q_length, k_length), QK^T.
    // attention_mask,     (batch_size, q_length, k_length), attention mask.
    int q_length = qk->shape[2];
    int batch_size = qk->shape[0];
    int head_nums = qk->shape[1];
    int k_length = qk->shape[3];
    dim3 grid(q_length, batch_size, head_nums);

    dim3 block((k_length + 32 - 1) / 32 * 32);//align with 32x threads
    if (block.x > 2048 && block.x <= 4096) {
        constexpr int NUMS_PER_THREAD_PER_ROW = 4;
        block.x /= 4;
        block.x = (block.x + 32 - 1) / 32 * 32;
        assert(block.x < 1024);
        ScaleMaskAndSoftmax<NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((float*)(attn_score->data),
                                                (float*)(qk->data),
                                                (uint8_t*)(mask->data),
                                                batch_size,
                                                head_nums,
                                                q_length,
                                                k_length,
                                                scale);      
    } else if (block.x > 1024) {
        constexpr int NUMS_PER_THREAD_PER_ROW = 2;
        block.x /= 2;
        block.x = (block.x + 32 - 1) / 32 * 32;
        assert(block.x < 1024);
        ScaleMaskAndSoftmax<NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((float*)(attn_score->data),
                                            (float*)(qk->data),
                                            (uint8_t*)(mask->data),
                                            batch_size,
                                            head_nums,
                                            q_length,
                                            k_length,
                                            scale);
    } else {
        constexpr int NUMS_PER_THREAD_PER_ROW = 1;
        assert(block.x < 1024);
        ScaleMaskAndSoftmax<NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((float*)(attn_score->data),
                                            (float*)(qk->data),
                                            (uint8_t*)(mask->data),
                                            batch_size,
                                            head_nums,
                                            q_length,
                                            k_length,
                                            scale);       
    }
}
