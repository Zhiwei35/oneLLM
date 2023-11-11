#include "src/kernels/softmax_kernel.h"
#include <float.h>
#define WarpSize 32
template<typename T>
struct Vec {};

template<>
struct Vec<half> {
   using Type = half2;
   static constexpr int size = 2;
};

template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};

template<typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a + b; }
};

template<typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(const T& a, const T& b) const { return max(a, b); }
};

template<template<typename> class ReductionOp, typename T, int warp_width = WarpSize>
__inline__ __device__ T WarpReduce(T val) {
  for (int mask = 32 / 2; mask > 0; mask /= 2) {
    val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
  }
  return val;
}

template<template<typename> class ReductionOp, typename T, int warp_width = WarpSize>
__device__ T blockReduce(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = blockDim.x / 32;
    __shared__ float warpmax[warpnum];
    val = WarpReduce<MaxOp, T>(val);
    if(laneid == 0){
        warpmax[wid] = val;
    }
    __syncthreads();

    T warpmax_reg = tid < warpnum ? warpmax[tid] : 0;
    final_max = WarpReduce<MaxOp, T>(warpmax_reg);
    return final_max;
}

template<typename float, typename T_QK>
__global__ void SoftmaxKernel(T* attention_score, // output, (batch_size, head_num, q_length, k_length)
                            T_QK* qk, //input, (batch_size, head_num, q_length, k_length)
                            const T* attention_mask, //input, (batch_size, q_length, k_length)
                            int batch_size,
                            int q_len,
                            int k_len,
                            int head_nums,
                            T qk_scale){
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    int batch_stride = batch_id * head_nums * q_len * k_len;
    int head_stride = head_id * q_len * k_len;
    using Vec_t = typename Vec<float>::Type;
    int vec_size = Vec<float>::size; 
    int vec_nums_per_block = blockDim.x;
    int vec_nums_per_row = (k_len + vec_size - 1)/ vec_size;
    int rows_per_block = 
    Vec_t* qk_vec = reinterpret_cast<Vec_t*>(qk);
    Vec_t qk_reg;
    float* qk_scalar;
    // float vec_sum; 
    float block_max[(q_len + gridDim.x - 1) / gridDim.x];
    for (int row_id = blockIdx.x; row_id < q_len; row_id += gridDim.x){
        float vec_max[row_id / q_len] = FLT_MIN;
        for (int vec_id = threadIdx.x; vec_id < vec_nums_per_row; vec_id += vec_nums_per_block) {
            int vec_offset = batch_stride + head_stride + row_id * k_len + vec_id * vec_size;
            qk_reg = *reinterpret_cast<Vec_t*>(&qk_vec[vec_offset]);
            qk_scalar = reinterpret_cast<float*>(qk_reg);
            // vec local max
            for (int i = 0; i < vec_size; i++) {
                vec_max[row_id / q_len] = max(vec_max[row_id], qk_scalar[i]);
            }
        }
        block_max[row_id / q_len] = blockReduce<MaxOp, float>(vec_max[row_id]);
    }
    __shared__ float block_sum[(q_len + gridDim.x - 1) / gridDim.x];
    for (int row_id = blockIdx.x; row_id < q_len; row_id += gridDim.x){
        for (int vec_id = threadIdx.x; vec_id < vec_nums_per_row; vec_id += vec_nums_per_block) {
            for (int i = 0; i < vec_size; i++) {
                block_sum[row_id / q_len] += expf(x - block_max[row_id / q_len]);
        }

    }

                                        }

template<typename half, typename T_QK>
__global__ void SoftmaxKernel(T* attention_score, // output, (batch_size, head_num, q_length, k_length)
                            T_QK* qk, //input, (batch_size, head_num, q_length, k_length)
                            const T* attention_mask, //input, (batch_size, q_length, k_length)
                            int batch_size,
                            int q_len,
                            int k_len,
                            int head_nums,
                            T qk_scale){
    int batch_id = blockIdx.y;
    int head_id = blockIdx.z;
    using Vec_t = typename Vec<half>::Type;
    int vec_size = Vec<half> ::size;
    for (int row_id = blockIdx.x; row_id < q_len; row_id += gridDim.x){
        for (int col_id = threadIdx.x; col <  )
    }

                                        }

template<typename T, typename T_QK>
void launchSoftmax(T* attention_score, // output, (batch_size, head_num, q_length, k_length)
                  T_QK* qk, //input, (batch_size, head_num, q_length, k_length)
                  const T* attention_mask, //input, (batch_size, q_length, k_length)
                  int batch_size,
                  int q_len,
                  int k_len,
                  int head_nums,
                  T qk_scale)
{
    // one block handle one row, row is k_len
    dim3 grid(q_len, batch_size, head_nums);
    while (q_len * batch_size * head_nums > 1024) {
        grid.x = q_len / 2;
    }
    // question: if k_len can't be divded by 4, ep:13, then the final ele can be still vectorized?
    int threads_num = (k_len + (Vec<T>::size - 1) / Vec<T>::size);
    threads_num = threads_num > 1024 ? threads_num / 4, threads_num;
    dim3 block(threads_num);
    std::cout << "in softmax, allocate " << q_len << " blockx, "
                                         << batch_size << " blocky, "
                                         << head_nums << " blockz, " 
                                         << threads_num << " threads." 
                                         << std::endl;
    SoftmaxKernel<T, T_QK><<<grid, block>>>(T* attention_score, // output, (batch_size, head_num, q_length, k_length)
                                            T_QK* qk, //input, (batch_size, head_num, q_length, k_length)
                                            const T* attention_mask, //input, (batch_size, q_length, k_length)
                                            int batch_size,
                                            int q_len,
                                            int k_len,
                                            int head_nums,
                                            T qk_scale);
}
