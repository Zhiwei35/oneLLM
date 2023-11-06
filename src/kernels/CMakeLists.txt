add_library(beamsearch_topK STATIC beamsearch_topK.cu)
set_property(TARGET beamsearch_topK PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET beamsearch_topK PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET beamsearch_topK PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(qkv_bias_and_rope STATIC qkv_bias_and_RoPE.cu)
set_property(TARGET qkv_bias_and_rope PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET qkv_bias_and_rope PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET qkv_bias_and_rope PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
