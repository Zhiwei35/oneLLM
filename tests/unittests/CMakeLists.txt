add_executable(topK
    test_beamsearch_topK.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  topK PUBLIC
    -lcudart
    -lcudadevrt
    beamsearch_topK)


add_executable(masked_attn
    test_decoder_masked_attn.cu
)
target_link_libraries(  # Libs for test_decoder_masked_attn
    masked_attn PUBLIC
    -lcudart
    -lcudadevrt
    decoder_masked_attn) 
