add_executable(embedding
    test_input_embedding.cu
)
target_link_libraries(  # Libs for test_softmax
    embedding PUBLIC
    -lcudart
    -lcudadevrt
    embeddingFunctor)
