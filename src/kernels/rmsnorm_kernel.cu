#include <stdio.h>
#include "src/kernels/rmsnorm_kernel.h"
//bugs1: 2nd warpreducesum returns 0, because blockDim.x < 32, blockDim.x / 32=0
//bugs2: output buffer valuse is the same as ones before call, thats because we didn't successfully write into the output address
//bugs3: output buffer's 1st 32 values are right, the latter is wrong, because when we use vec, the ele nums of a row is hiddenunits/vecsize, we should note the row stride to move the ptr carefully
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
    for(int i = 32 / 2; i > 0; i >>= 1){
        val += __shfl_xor_sync(0xffffffff, val, i);
    }
    return val; // 32 threads return val, but only 0th thread is sum val
}
//note:!!!when blocksize < 32, use blockDim.x/32 to get warp nums is wrong, we should instead ceil it
template<typename T>
__device__ T blockReduceSum(T val){
    int tid = threadIdx.x;
    int wid = tid / 32;
    int laneid = tid % 32;
    int warpnum = (blockDim.x + 31) / 32;
    static __shared__ T warpsum[64];
    val = warpReduceSum<T>(val);
    if(laneid == 0){
        warpsum[wid] = val;
    }
    __syncthreads();

    T sum = tid < warpnum ? warpsum[tid] : 0;
    sum = warpReduceSum<T>(sum); //though 0th own the sum, but dont need to shfl sync
    return sum;
}
// 1.this kernel is used after self attention and FFN in every layer
// 2.I allocate threads number by assuming head size can be divided by 4 and 2
__global__ void RMSNorm(float* decoder_out, // [num tokens, q_hidden_units]
                        float* scale, //[q_hidden_units], RMSNorm weights
                        float eps, //RMSNorm eps
                        int num_tokens, 
                        int hidden_units){
    int vec_size = Vec<float>::size;
    using Vec_t = typename Vec<float>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t* s;
    Vec_t dout, tmp;
    
    float thread_accm = static_cast<float>(0);

    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        dout = reinterpret_cast<Vec_t*>(decoder_out)[batch_id * hidden_units / vec_size + i];// note the offset should divide vec size
        thread_accm += dout.x * dout.x + dout.y * dout.y + 
                       dout.z * dout.z + dout.w * dout.w;
    } //x^2
    
    // mean(x^2)
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){
        inv_fenmu = rsqrt(blocksum / hidden_units + eps);
    }
    // rmsnorm
    Vec_t* out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);// note before vec the stride is batch_id * hiddenunits w/o / vecsize
    s = reinterpret_cast<Vec_t*>(const_cast<float*>(scale));
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        //s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale))[i];
        out[i].x = s[i].x * out[i].x * inv_fenmu;
        out[i].y = s[i].y * out[i].y * inv_fenmu;
        out[i].z = s[i].z * out[i].z * inv_fenmu;
        out[i].w = s[i].w * out[i].w * inv_fenmu;
    }    
}

void launchRMSNorm( Tensor* decoder_out, // [num tokens, hidden_units]
                    LayerNormWeight& attn_norm_weight, //RMSNorm weights
                    float eps //RMSNorm eps
                    )
{
    int num_tokens = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    int vec_size = Vec<float>::size;
    int num_threads = hidden_units / vec_size; // assume head size can be divided by 4 and 2
    dim3 grid(num_tokens);
    dim3 block(num_threads);
    printf("calling RMSNorm\n");
    RMSNorm<<<grid, block>>>((float*)decoder_out->data,
                            attn_norm_weight.gamma,
                            eps,
                            num_tokens,
                            hidden_units);
    printf("called RMSNorm\n");
}
