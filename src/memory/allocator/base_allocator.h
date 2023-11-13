#include <cuda.h>
#include <cuda_runtime.h>

class BaseAllocator 
{
    virtual ~BaseAllocator(){};
    // unified interface for alloc buffer
    template<typename T>
    T* Malloc(T* ptr, size_t size, bool is_host){
        return deviceMalloc(ptr, size, is_host);
    }
    template<typename T>
    virtual T* deviceMalloc(T* ptr, size_t size, bool is_host = false) = 0;
    template<typename T>
    virtual void deviceFree(T* ptr, bool is_host = false) const = 0;
}
