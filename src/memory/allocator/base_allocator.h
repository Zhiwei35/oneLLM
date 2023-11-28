#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator 
{
public:
    //BaseAllocator() = default;//class with pure virtual func cant be initialized
    virtual ~BaseAllocator(){};
    // unified interface for alloc buffer
    template<typename T>
    void* Malloc(T* ptr, size_t size, bool is_host){
        return deviceMalloc((void*)ptr, size, is_host);
    }
    virtual void* deviceMalloc(void* ptr, size_t size, bool is_host = false) = 0;
    virtual void deviceFree(void* ptr, bool is_host = false) = 0;
};
