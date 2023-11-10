#include <cuda_runtime.h>
#include <cuda.h>

template<typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    T* residual, 
                                    T* decoder_out, // [num tokens, hidden_units]
                                    const T* bias,
                                    const T* scale, //RMSNorm weights
                                    float eps, //RMSNorm eps
                                    int num_tokens, 
                                    int hidden_units);