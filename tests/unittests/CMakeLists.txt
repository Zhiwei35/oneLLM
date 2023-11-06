add_executable(unittest
    test_beamsearch_topK.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  unittest PUBLIC
    -lcudart
    -lcudadevrt
    beamsearch_topK) 

add_executable(biasRope
    test_qkv_bias_and_rope.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  biasRope PUBLIC
    -lcudart
    -lcudadevrt
    qkv_bias_and_rope) 