add_executable(unittest
    test_beamsearch_topK.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  unittest PUBLIC
    -lcudart
    -lcudadevrt
    beamsearch_topK) 

add_executable(causalmask
    test_causal_mask.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
causalmask PUBLIC
    -lcudart
    -lcudadevrt
    build_casual_mask) 