#include <cuda_runtime.h>
template<typename T>
void launchAct(const T* input, T* out, const int intermedia_size);