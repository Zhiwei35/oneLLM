add_executable(unittest
    test_beamsearch_topK.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  unittest PUBLIC
    -lcudart
    -lcudadevrt
    beamsearch_topK) 

add_executable(paddingoffset
    test_cal_paddingoffset.cu
)
target_link_libraries(  # Libs for test_cal_paddingoffset
  paddingoffset PUBLIC
    -lcudart
    -lcudadevrt
    cal_paddingoffset) 

add_executable(causalmask
    test_casual_mask.cu
)
target_link_libraries(  # Libs for test_build_casual_mask
causalmask PUBLIC
    -lcudart
    -lcudadevrt
    build_casual_mask) 

add_executable(appendkv
    test_append_kvcache.cu
)
target_link_libraries(  # Libs for test_append_kvcache
    appendkv PUBLIC
    -lcudart
    -lcudadevrt
    appendtokvcache) 

add_executable(linear
    test_linear.cu
)
target_link_libraries(  # Libs for test_linear
  linear PUBLIC
    -lcudart
    -lcudadevrt
    qkv_linear) 

