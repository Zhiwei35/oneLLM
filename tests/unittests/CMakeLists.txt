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