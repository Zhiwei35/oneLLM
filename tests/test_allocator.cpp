#include "src/memory/allocator/cuda_allocator.h"
//expected behavior:
//alloc 1 small block, 9 big block
//10 blocks is set is_allocated=false
void test1(){
    BaseAllocator* allocator = new CudaAllocator;
    int nums = 10;
    float* ptr_arrays[nums];
    for (int i = 0; i < nums; i++) {
        float* ptr = ptr_arrays[i];
        size_t size = sizeof(float) * (i + 250) * 1024;//前几个小block,后几个大block
        ptr = deviceMalloc(ptr, size, false);
    }
    for (int i = 0; i < nums; i++) {
        float* ptr = ptr_arrays[i];
        deviceFree(ptr);
    }
    for (int i = 0; i < nums; i++) {
        float* ptr = ptr_arrays[i];
        cudaFree(ptr);
    }
    delete []ptr_arrays;
}
//expected behavior:
//alloc 2048 small block
//some blocks is set is_allocated=false to some extent, then free the suipian
void test2(){
    BaseAllocator* allocator = new CudaAllocator;
    int nums = 2048;
    float* ptr_arrays[nums];
    for (int i = 0; i < nums; i++) {
        float* ptr = ptr_arrays[i];
        size_t size = sizeof(float) * 128 * 1024;//前几个小block,后几个大block
        ptr = deviceMalloc(ptr, size, false);
    }
    for (int i = 0; i < nums; i++) {
        float* ptr = ptr_arrays[i];
        deviceFree(ptr);
    }
    for (int i = 0; i < nums; i++) {
        float* ptr = ptr_arrays[i];
        cudaFree(ptr);
    }
    delete []ptr_arrays;
}

int main(){
    test1();
    test2();
}