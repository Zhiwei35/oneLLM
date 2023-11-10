add_executable(unittest
    test_beamsearch_topK.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  unittest PUBLIC
    -lcudart
    -lcudadevrt
    beamsearch_topK) 


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

add_executable(fused_addres_rms
    test_fused_addres_rms.cu
)
target_link_libraries(  # Libs for test_addres_rmsnorm
    fused_addres_rms PUBLIC
    -lcudart
    -lcudadevrt
    addresandrms) 