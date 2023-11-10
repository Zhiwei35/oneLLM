add_library(beamsearch_topK STATIC beamsearch_topK.cu)
set_property(TARGET beamsearch_topK PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET beamsearch_topK PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET beamsearch_topK PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(appendtokvcache STATIC append_to_kvcache.cu)
set_property(TARGET appendtokvcache PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET appendtokvcache PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET appendtokvcache PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(cublasWrapper STATIC cublas_wrapper.cc)
set_property(TARGET cublasWrapper PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cublasWrapper PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(qkv_linear STATIC qkv_linear.cu)
set_property(TARGET qkv_linear PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET qkv_linear PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET qkv_linear PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(qkv_linear PUBLIC -lcudart -lcublas cublasWrapper)

add_library(addresandrms STATIC fused_addresidual_norm.cu)
set_property(TARGET addresandrms PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET addresandrms PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET addresandrms PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

