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
target_link_libraries(  # Libs for test_beamsearch_kernel
    appendkv PUBLIC
    -lcudart
    -lcudadevrt
    appendtokvcache) 
