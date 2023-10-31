add_executable(topK
    test_beamsearch_topK.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  topK PUBLIC
    -lcudart
    -lcudadevrt
    beamsearch_topK)

add_executable(act
    test_activation.cu
)
target_link_libraries(  # Libs for test_beamsearch_kernel
  act PUBLIC
    -lcudart
    -lcudadevrt
    activation)

