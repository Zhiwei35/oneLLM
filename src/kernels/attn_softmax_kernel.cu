#include "src/kernels/attn_softmax_kernel.h"
#include <float.h>
#include <math.h>
#include <assert.h>
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
    // you can change L61 with __shfl_down_sync like 6_warp_level_reduce and see performance change
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
    float warp_val = tid < warp_nums ? warp[warp_id] : 0;
    return warpReduce<ReductionOp, T>(warp_val);
}


template<typename T, int NUMS_PER_THREAD_PER_ROW>
__global__ void ScaleMaskAndSoftmax(T* attn_score,
                                    T* qk,
                                    T* mask,
                                    int batch_size,
                                    int head_nums,
                                    int q_len,
                                    int k_len,
                                    float scale)
{
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    //const int NUMS_PER_THREAD_PER_ROW = ceil(k_len / blockDim.x);
    __shared__ T inv_sum, s_max;
    
    for(int row_start = blockIdx.x; row_start < q_len; row_start += gridDim.x) {
        int qk_offset = 0;
        int mask_offset = 0;
        T qk_data = static_cast<T>(0);
        T mask_data = static_cast<T>(0);
        T thread_max = FLT_MIN;
        T data[NUMS_PER_THREAD_PER_ROW]; // 面对这种一个block一个thread需要处理多行多列的时候，数据尽量用数组存储，计算出每个block和thread要处理几行几列
        //for(int col_start = threadIdx.x; col_start < k_len; col_start += blockDim.x){
        for(int col_start = 0; col_start < NUMS_PER_THREAD_PER_ROW; col_start++){   
            qk_offset = batch_id * head_nums * q_len * k_len + head_id * q_len * k_len
                            + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            qk_data = qk[qk_offset];
            mask_offset = batch_id * q_len * k_len + row_start * k_len + col_start * blockDim.x + threadIdx.x;
            mask_data = mask[mask_offset];
            data[col_start] = (T)(scale) * qk_data + mask_data;
            thread_max = max(data[col_start], thread_max);
        }
        // warp/block reduce
        T max_val = blockReduce<MaxOp, T>(thread_max);
        if (threadIdx.x == 0) {
            s_max = max_val;
        }
        __syncthreads();
        // thread local fenzi/fenmu
        T thread_sum;
        for(int col_start = threadIdx.x; col_start < k_len; col_start += blockDim.x){
            data[col_start / blockDim.x] = exp(data[col_start / blockDim.x] - s_max);
            thread_sum += data[col_start / blockDim.x];
        }
        // row sum
        T sum = blockReduce<SumOp, T>(thread_sum);
        if(threadIdx.x == 0) {
            inv_sum = 1 / sum;//maybe sum(fenmu) need to add a small value to keep stable
        }
        __syncthreads();
        // write back into gmem
        for(int col_start = threadIdx.x; col_start < k_len; col_start += blockDim.x){
            qk_offset = batch_id * head_nums * q_len * k_len + head_id * q_len * k_len
                            + row_start * k_len + col_start;
            attn_score[qk_offset] = (data[col_start / blockDim.x] * inv_sum);
        }
    }
}

template<typename T, typename T1>
void launchScaleMaskAndSoftmax(Tensor<T>* qk,
                               Tensor<T1>* mask,
                               Tensor<T>* attn_score,
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
        ScaleMaskAndSoftmax<T, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((T*)attn_score->data,
                                                (T*)qk->data,
                                                (T1*)mask->data,
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
        ScaleMaskAndSoftmax<T, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((T*)attn_score->data,
                                            (T*)qk->data,
                                            (T*)mask->data,
                                            batch_size,
                                            head_nums,
                                            q_length,
                                            k_length,
                                            scale);
    } else {
        constexpr int NUMS_PER_THREAD_PER_ROW = 1;
        assert(block.x < 1024);
        ScaleMaskAndSoftmax<T, NUMS_PER_THREAD_PER_ROW><<<grid, block>>>((T*)attn_score->data,
                                            (T*)qk->data,
                                            (T*)mask->data,
                                            batch_size,
                                            head_nums,
                                            q_length,
                                            k_length,
                                            scale);       
    }
}

template void launchScaleMaskAndSoftmax(Tensor<float>* qk,
                               Tensor<uint8_t>* mask,
                               Tensor<float>* attn_score,
                               float scale)