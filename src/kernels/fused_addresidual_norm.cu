#include <stdio.h>
#include "src/kernels/fused_addresidual_norm.h"
//bugs1: 2nd warpreducesum returns 0, because blockDim.x < 32, blockDim.x / 32=0
//bugs2: output buffer valuse is the same as ones before call, thats because we didn't successfully write into the output address
//bugs3: output buffer's 1st 32 values are right, the latter is wrong, because when we use vec, the ele nums of a row is hiddenunits/vecsize, we should note the row stride to move the ptr carefully

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

    T sum = tid < warpnum ? warpsum[tid] : (T)0.0f;
//    printf("tid=%d, blocksize=%d, warpnum=%d,sum=%f\n",tid, blockDim.x, warpnum, sum);
    sum = warpReduceSum<T>(sum); //though 0th own the sum, but dont need to shfl sync
//    if(tid == 0){
//        printf("tid=0,sum=%f, warpsum[0]=%f\n",sum,warpsum[0]);
//    }
    return sum;
}
// 1.this kernel is used after self attention and FFN in every layer
// 2.I allocate threads number by assuming head size can be divided by 4 and 2
template<typename T>
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
    Vec_t *rsd, *bia, *s;
    Vec_t dout, tmp;
    if(batch_id == 0 && tid == 0) {
        printf("ctx attn output: \n");
        printf("%f\n",decoder_out[0]);
        printf("%f\n",decoder_out[1]);
    }
   // printf("in kernel\n");    
    T thread_accm = static_cast<T>(0);
    if (residual != nullptr && bias != nullptr){
        rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);//note the offset     should divide vec size
        bia = reinterpret_cast<Vec_t*>(const_cast<T*>(bias));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        //if (residual != nullptr && bias != nullptr){
           // rsd = reinterpret_cast<Vec_t*>(residual)[batch_id * hidden_units / vec_size + i];//note the offset should divide vec size
          //  bia = reinterpret_cast<Vec_t*>(const_cast<T*>(bias))[i];
        //}
        dout = reinterpret_cast<Vec_t*>(decoder_out)[batch_id * hidden_units / vec_size + i];// note the offset should divide vec size
        tmp.x = dout.x + rsd[i].x + bia[i].x;
        tmp.y = dout.y + rsd[i].y + bia[i].y;
        tmp.z = dout.z + rsd[i].z + bia[i].z;
        tmp.w = dout.w + rsd[i].w + bia[i].w;
        thread_accm += tmp.x * tmp.x + tmp.y * tmp.y + 
                       tmp.z * tmp.z + tmp.w * tmp.w;
    } // addresidual
  //  printf("in kernel 1\n");
    // mean(x^2)
    T blocksum = blockReduceSum<T>(thread_accm);
    __shared__ float inv_fenmu;
    if(tid == 0){
        //debug info printf("blocksum on GPU is %f\n", blocksum);
        inv_fenmu = rsqrt(blocksum / hidden_units + eps);
        //debug info printf("inv_fenmu on GPU is %f\n", inv_fenmu);
    }
    // rmsnorm
    Vec_t* out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);// note before vec the stride is batch_id * hiddenunits w/o / vecsize
    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        //s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale))[i];
        out[i].x = s[i].x * out[i].x * inv_fenmu;
        out[i].y = s[i].y * out[i].y * inv_fenmu;
        out[i].z = s[i].z * out[i].z * inv_fenmu;
        out[i].w = s[i].w * out[i].w * inv_fenmu;
        if(i == 0) {
            printf("ctx attn residual rmsnorm top2 res: \n");
            printf("out.x = %f, s[i].x = %f, inv_fenmu.x = %f\n",out[i].x, s[i].x, inv_fenmu);
            printf("out.y = %f, s[i].y = %f, inv_fenmu.y = %f\n",out[i].y, s[i].y);
        }
    } 

//    printf("in kernel 2\n");
}

template<>
__global__ void FusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    half* residual, 
                                    half* decoder_out, // [num tokens, hidden_units]
                                    const half* bias, //[hidden_units]
                                    const half* scale, //[hidden_units], RMSNorm weights
                                    float eps, //RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units){
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int batch_id = blockIdx.x;
    int tid = threadIdx.x;
    Vec_t *rsd, *bia, *s;
    Vec_t dout, tmp;
   // printf("in kernel\n");    
    float thread_accm = 0.0f;
    if (residual != nullptr && bias != nullptr){
        rsd = reinterpret_cast<Vec_t*>(residual + batch_id * hidden_units);//note the offset     should divide vec size
        bia = reinterpret_cast<Vec_t*>(const_cast<half*>(bias));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        dout = reinterpret_cast<Vec_t*>(decoder_out)[batch_id * hidden_units / vec_size + i];// note the offset should divide vec size
        tmp = __hadd2(__hadd2(dout, rsd[i]), bia[i]);
        thread_accm += __half2float(tmp.x) * __half2float(tmp.x) + __half2float(tmp.y) * __half2float(tmp.y);
    } // addresidual
  //  printf("in kernel 1\n");
    // mean(x^2)
    float blocksum = blockReduceSum<float>(thread_accm);
    __shared__ Vec_t inv_fenmu;
    if(tid == 0){
        //debug info printf("blocksum on GPU is %f\n", blocksum);
        inv_fenmu = scalar_cast_vec<Vec_t>(__float2half(rsqrt(blocksum / hidden_units + eps)));
        //debug info printf("inv_fenmu on GPU is %f\n", inv_fenmu);
    }
    // rmsnorm
    Vec_t* out = reinterpret_cast<Vec_t*>(decoder_out + batch_id * hidden_units);// note before vec the stride is batch_id * hiddenunits w/o / vecsize
    if (scale != nullptr){
        s = reinterpret_cast<Vec_t*>(const_cast<half*>(scale));
    }
    for(int i = tid; i < hidden_units / vec_size; i += blockDim.x) {
        //s = reinterpret_cast<Vec_t*>(const_cast<T*>(scale))[i];
        out[i] = __hmul2(__hmul2(s[i], out[i]), inv_fenmu);
    } 
//    printf("in kernel 2\n");
}

template<typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    TensorWrapper<T>* residual, 
                                    TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<T>& norm,
                                    T* scale, //RMSNorm weights
                                    float eps) //RMSNorm eps
{
    int batch_size = decoder_out->shape[0];
    int hidden_units = decoder_out->shape[1];
    T* bias = norm.bias;
    T* gamma = scale;
    int vec_size = Vec<T>::size;
    int num_threads = hidden_units / vec_size; // assume head size can be divided by 4 and 2
    dim3 grid(batch_size);
    dim3 block(num_threads);
    // printf("calling fusedAddBiasResidualAndRMSNorm\n");
    FusedAddBiasResidualRMSNorm<T><<<grid, block>>>(residual->data, 
                                                decoder_out->data,
                                                bias,
                                                gamma,
                                                eps,
                                                batch_size,
                                                hidden_units);
    // printf("called fusedAddBiasResidualAndRMSNorm\n");
}
template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    TensorWrapper<float>* residual, 
                                    TensorWrapper<float>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<float>& norm,
                                    float* scale, //RMSNorm weights
                                    float eps);
template void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    TensorWrapper<half>* residual, 
                                    TensorWrapper<half>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<half>& norm,
                                    half* scale, //RMSNorm weights
                                    float eps);
