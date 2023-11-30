cmake_minimum_required(VERSION 3.8)

add_library(LlamaCtxAttn STATIC context_attention.cpp)
set_property(TARGET LlamaCtxAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaCtxAttn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaCtxAttn PUBLIC
#                             -lcudart
                             -lcudadevrt
                             qkv_bias_and_rope
                             appendtokvcache
#                             cublasWrapper
                             qkv_linear
                             fused_transpose_reshape_remv_pad
                             transpose
                             softmax)

add_library(LlamaselfAttn STATIC masked_self_attention.cpp)

set_property(TARGET LlamaselfAttn PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaselfAttn PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaselfAttn PUBLIC
                             -lcudart
                             -lcudadevrt
                             decoder_masked_attn
)
