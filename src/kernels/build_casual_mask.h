#include<cuda_runtime.h>
#include<cuda.h>

template<typename T>
void launchBuildCausalMasks(T* mask, 
                            const int* q_lens, 
                            const int* k_lens, 
                            int max_q_len, 
                            int max_k_len, 
                            int batch_size);