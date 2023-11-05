#include "cublas_wrapper.h"
#include <cuda_runtime.h>
#include <cuda.h>

template<typename T>
void launchLinear(const T* input,
                  T* output, 
                  const int input_2nd_dim, 
                  const T* weight,
                  const int hidden_units);

