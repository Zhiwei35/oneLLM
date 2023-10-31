#include <iostream>
#include "src/kernels/activation_kernel.h"

template<typename T>
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}

template<typename T>
__global__ void silu_and_mul_kernel(
  T* __restrict__ out,               // [bs, intermedia size]
  const T* __restrict__ input,       // [bs, 2, intermedia size]
  const int intermedia_size) {
  const int batch_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < intermedia_size; idx += blockDim.x) {
    const T x = input[batch_idx * 2 * intermedia_size + idx];
    const T y = input[batch_idx * 2 * intermedia_size + intermedia_size + idx];
    out[batch_idx * intermedia_size + idx] = silu<T>(x) * y;
  }
}

template<typename T>
void launchAct(const T* input, T* out, const int batch_size, const int intermedia_size) {
    dim3 grid(batch_size);
    dim3 block(256);
    silu_and_mul_kernel<T><<<grid, block>>>(out, input, intermedia_size);
}

template void launchAct(const float* input, float* output, const int batch_size, const int intermedia_size);
