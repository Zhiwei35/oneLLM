add_library(cudautils STATIC cuda_utils.cu)
set_property(TARGET cudautils PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cudautils PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cudautils PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
