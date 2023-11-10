#include <iostream>
#include "src/kernels/fused_addresidual_norm.h"

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
__device__ T warpReduceSum(T val){
    for(int i = 32 / 2; i > 0; i >>= 2){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 32 threads return val, but only 0th thread is sum val
}
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = blockDim.x / 32;
    __shared__ float warpsum[warpnum];
    val = warpReduceSum<T>(val);
    if(laneid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();

    T sum = tid < warpnum ? warpsum[tid] : 0;
    sum = warpReduceSum<T>(sum); //though 0th own the sum, but dont need to shfl sync
    //sum = __shfl_sync(0xffffffff, sum, 0);
    return sum;
}
// 1.this kernel is used after self attention and FFN in every layer
// 2.I allocate threads number by assuming head size can be divided by 4 and 2
__global__ void FusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    T* residual, 
                                    T* decoder_out, // [num tokens, hidden_units]
                                    const T* bias, //[hidden_units]
                                    const T* scale, //[hidden_units], RMSNorm weights
                                    float eps, //RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units){
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t rsd, bia, dout, s;
    
    T thread_accm = static_cast<T>(0);
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        rsd = reinterpret_cast<Vec_t*>(residual)[batch_id * hidden_units + i];
        bia = reinterpret_cast<Vec_t*>(bias)[i];
        dout = reinterpret_cast<Vec_t*>(decoder_out)[batch_id * hidden_units + i];
        dout.x += rsd.x + bia.x;
        dout.y += rsd.y + bia.y;
        dout.z += rsd.z + bia.z;
        dout.w += rsd.w + bia.w;
        thread_accm += dout.x * dout.x + dout.y * dout.y + 
                       dout.z * dout.z + dout.w * dout.w;
    } // addresidual

    // mean(x^2)
    T blocksum = blockReduceSum<T>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){
        inv_fenmu = rsqrt(blocksum / hidden_units + eps);
    }
    // rmsnorm
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        dout = reinterpret_cast<Vec_t*>(decoder_out)[batch_id * hidden_units + i];
        s = reinterpret_cast<Vec_t*>(scale)[i];
        dout.x = s.x * dout.x * inv_fenmu;
        dout.y = s.y * dout.y * inv_fenmu;
        dout.z = s.z * dout.z * inv_fenmu;
        dout.w = s.w * dout.w * inv_fenmu;
    }
}

template<typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    T* residual, 
                                    T* decoder_out, // [num tokens, hidden_units]
                                    const T* bias,
                                    const T* scale, //RMSNorm weights
                                    float eps, //RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units)
{
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / vec_size; // assume head size can be divided by 4 and 2
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    FusedAddBiasResidualRMSNorm<<<grid, block>>>(residual, 
                                                decoder_out,
                                                bias,
                                                scale,
                                                eps,
                                                num_tokens,
                                                hidden_units);
}

template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    float* residual, 
                                    float* decoder_out, // [num tokens, hidden_units]
                                    const float* bias,
                                    const float* scale, //RMSNorm weights
                                    float eps, //RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units);