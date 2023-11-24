cmake_minimum_required(VERSION 3.8)

add_library(LlamaCtxdecoder STATIC context_decoder.cpp)
set_property(TARGET LlamaCtxdecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET LlamaCtxdecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(LlamaCtxdecoder PUBLIC
                                        LlamaCtxAttn
                                        Llamaffn 
                                        llamaweights
                                        cal_paddingoffset
                                        build_casual_mask
                                        addresandrms
)
