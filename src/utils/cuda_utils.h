#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>
#include <iostream>
#include "src/utils/macro.h"

template<typename T>
void GPUMalloc(T** ptr, size_t size);

template<typename T>
void GPUFree(T* ptr);

template<typename T, typename T_IN>
int loadWeightFromBin(T* ptr, std::vector<size_t> shape, std::string filename);
