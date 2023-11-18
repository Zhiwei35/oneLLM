# add_executable(unittest
#     test_beamsearch_topK.cu
# )
# target_link_libraries(  # Libs for test_beamsearch_kernel
#   unittest PUBLIC
#     -lcudart
#     -lcudadevrt
#     beamsearch_topK) 

add_executable(biasRope
     test_qkv_bias_and_rope.cu
)
target_link_libraries(  # Libs for test_qkv_bias_and_rope
   biasRope PUBLIC
     -lcudart
     -lcudadevrt
     qkv_bias_and_rope) 

# add_executable(paddingoffset
#     test_cal_paddingoffset.cu
# )
# target_link_libraries(  # Libs for test_cal_paddingoffset
#   paddingoffset PUBLIC
#     -lcudart
#     -lcudadevrt
#     cal_paddingoffset) 

# add_executable(causalmask
#     test_casual_mask.cu
# )
# target_link_libraries(  # Libs for test_build_casual_mask
#    causalmask PUBLIC
#     -lcudart
#     -lcudadevrt
#     build_casual_mask) 

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
add_executable(bmm
    test_batch_gemm.cu
)
target_link_libraries(  # Libs for test_linear
  bmm PUBLIC
    -lcudart
    -lcudadevrt
    qkv_linear)

add_executable(test_trans
    test_transpose.cu
)
target_link_libraries(  # Libs for test_linear
  test_trans PUBLIC
    -lcudart
    -lcudadevrt
    transpose)
add_executable(fused_trans_remv_pad
    test_fused_trans_remv_pad.cu
)
target_link_libraries(  # Libs for test_linear
  fused_trans_remv_pad PUBLIC
    -lcudart
    -lcudadevrt
    fused_transpose_reshape_remv_pad)
     

add_executable(attn_softmax
    test_softmax.cu
)
target_link_libraries(  # Libs for test_softmax
    attn_softmax PUBLIC
    -lcudart
    -lcudadevrt
    softmax) 
