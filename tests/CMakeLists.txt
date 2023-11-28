add_subdirectory(unittests)

#add_executable(test_tensor
#    test_tensor.cpp
#)
add_executable(test_alloc
    test_allocator.cpp
)
target_link_libraries(
  test_alloc PUBLIC
    -lcudart
    -lcudadevrt) 
