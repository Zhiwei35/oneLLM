#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
template<typename T>
void launchFusedAddBiasResidualRMSNorm( // residual.shape = [num tokens, hidden_units], batch_size = num tokens, n_dims = hidden_units
                                    TensorWrapper<T>* residual, 
                                    TensorWrapper<T>* decoder_out, // [num tokens, hidden_units]
                                    BaseWeight<T>& norm,
                                    LayerNormWeight<T>& scale, //RMSNorm weights
                                    float eps);
