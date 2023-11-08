add_executable(unittest
    test_beamsearch_topK.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  unittest PUBLIC
    -lcudart
    -lcudadevrt
    beamsearch_topK) 

add_executable(linear
    test_linear.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  linear PUBLIC
    -lcudart
    -lcudadevrt
    qkv_linear) 
