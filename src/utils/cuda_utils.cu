#include "src/utils/cuda_utils.h"

template<typename T>
void GPUMalloc(T** ptr, size_t size)
{
    ONELLM_CHECK_WITH_INFO(size >= ((size_t)0), "Ask deviceMalloc size " + std::to_string(size) + "< 0 is invalid.");
    CHECK(cudaMalloc((void**)(ptr), sizeof(T) * size));
}

template void GPUMalloc(float** ptr, size_t size);

template<typename T>
void GPUFree(T*& ptr)
{
    if (ptr != NULL) {
        CHECK(cudaFree(ptr));
        ptr = NULL;
    }
}

template void GPUFree(float*& ptr);

template<typename T>
void cudaH2Dcpy(T* tgt, const T* src, const size_t size)
{
    CHECK(cudaMemcpy(tgt, src, sizeof(T) * size, cudaMemcpyHostToDevice));
}

template void cudaH2Dcpy(float* tgt, const float* src, const size_t size);
// template<typename T_IN, typename T_OUT>
// __global__ void cudaD2DcpyConvert(T_OUT* dst, const T_IN* src, const size_t size)
// {
//     for (size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < size; tid += blockDim.x * gridDim.x) {
//         dst[tid] = cuda_cast<T_OUT>(src[tid]);
//     }
// }

// template<typename T_IN, typename T_OUT>
// void invokeCudaD2DcpyConvert(T_OUT* tgt, const T_IN* src, const size_t size)
// {
//     cudaD2DcpyConvert<<<256, 256, 0, 0>>>(tgt, src, size);
// }

// from FT code
// loads data from binary file. If it succeeds, returns a non-empty (shape size) vector. If loading fails or
// the product of the elements in shape is 0, this function will return an empty vector.
template<typename T>
std::vector<T> loadWeightFromBinHelper(std::vector<size_t> shape, std::string filename)
{
    if (shape.size() > 2) {
        printf("[ERROR] shape should have less than two dims \n");
        return std::vector<T>();
    }
    size_t dim0 = shape[0], dim1 = 1;
    if (shape.size() == 2) {
        dim1 = shape[1];
    }
    size_t size = dim0 * dim1;
    if (size == 0) {
        std::cout << "shape is zero, skip loading weight from file: " << filename << std::endl;
        return std::vector<T>();
    }

    std::vector<T> host_array(size);
    std::ifstream  in(filename, std::ios::in | std::ios::binary);
    if (!in.is_open()) {
        std::cout << "file" << filename << "cannot be opened, loading model fails!" << std::endl;
        return std::vector<T>();
    }

    size_t loaded_data_size = sizeof(T) * size;
    in.seekg(0, in.end);
    in.seekg(0, in.beg);

    std::cout << "Read " << std::to_string(loaded_data_size) << " bytes from " << filename << std::endl;
    in.read((char*)host_array.data(), loaded_data_size);

    size_t in_get_size = in.gcount();
    if (in_get_size != loaded_data_size) {
        return std::vector<T>();
    }
    in.close();
    // If we succeed, return an array with values.
    return host_array;
}

template<typename T, typename T_IN>
int loadWeightFromBin(T* ptr, std::vector<size_t> shape, std::string filename)
{
    std::vector<T_IN> host_array = loadWeightFromBinHelper<T_IN>(shape, filename);

    if (host_array.empty()) {
        return 0;
    }

    if (std::is_same<T, T_IN>::value == true) {
        cudaH2Dcpy(ptr, (T*)host_array.data(), host_array.size());
    }
    // TODO: add ptx type conversion later
    // else {
    //     T_IN* ptr_2 = nullptr;
    //     GPUMalloc(&ptr_2, host_array.size());
    //     cudaH2Dcpy(ptr_2, host_array.data(), host_array.size());
    //     invokeCudaD2DcpyConvert(ptr, ptr_2, host_array.size());
    //     GPUFree(ptr_2);
    // }
    return 0;
}

template int loadWeightFromBin<float, float>(float* ptr, std::vector<size_t> shape, std::string filename);
template int loadWeightFromBin<half, half>(half* ptr, std::vector<size_t> shape, std::string filename);
