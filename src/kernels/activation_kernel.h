#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
template<typename T>
void launchAct(const TensorWrapper<T>* input, TensorWrapper<T>* out);