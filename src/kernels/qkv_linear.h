#include <cuda_runtime.h>
#include <cuda.h>
#include "src/kernels/cublas_wrapper.h"
//template<typename T>
void launchLinear(const float* input,
                  float* output, 
                  const int input_2nd_dim, 
                  const float* weight,
                  const int hidden_units);

