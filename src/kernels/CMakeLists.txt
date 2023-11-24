
add_library(beamsearch_topK STATIC beamsearch_topK.cu)
set_property(TARGET beamsearch_topK PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET beamsearch_topK PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET beamsearch_topK PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(activation STATIC activation_kernel.cu)
set_property(TARGET activation PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET activation PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET activation PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(decoder_masked_attn STATIC decoder_masked_attn.cu)
set_property(TARGET decoder_masked_attn PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET decoder_masked_attn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET decoder_masked_attn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

# add_library(beamsearch_topK STATIC beamsearch_topK.cu)
# set_property(TARGET beamsearch_topK PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
# set_property(TARGET beamsearch_topK PROPERTY POSITION_INDEPENDENT_CODE  ON)
# set_property(TARGET beamsearch_topK PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(qkv_bias_and_rope STATIC qkv_bias_and_RoPE.cu)
set_property(TARGET qkv_bias_and_rope PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET qkv_bias_and_rope PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET qkv_bias_and_rope PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(cal_paddingoffset STATIC cal_paddingoffset.cu)
set_property(TARGET cal_paddingoffset PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET cal_paddingoffset PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET cal_paddingoffset PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(build_casual_mask STATIC build_casual_mask.cu)
set_property(TARGET build_casual_mask PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET build_casual_mask PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET build_casual_mask PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

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

add_library(fused_transpose_reshape_remv_pad STATIC fused_transpose_reshape_remv_pad.cu)
set_property(TARGET fused_transpose_reshape_remv_pad PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET fused_transpose_reshape_remv_pad PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET fused_transpose_reshape_remv_pad PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(transpose STATIC transpose_kernel.cu)
set_property(TARGET transpose PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET transpose PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET transpose PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(softmax STATIC attn_softmax_kernel.cu)
set_property(TARGET softmax PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET softmax PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET softmax PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)

add_library(embeddingFunctor STATIC input_embedding.cu)
set_property(TARGET embeddingFunctor PROPERTY CUDA_SEPARABLE_COMPILATION   ON)
set_property(TARGET embeddingFunctor PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET embeddingFunctor PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
