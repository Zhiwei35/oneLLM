#pragma once
#include <vector>
#include <cuda_fp16.h>
enum class WeightType
{
    FP32_W,
    FP16_W,
    INT8_W,
    UNSUPPORTED_W
};

inline int getBitNum(WeightType type)
{
    switch (type) {
        case WeightType::FP32_W:
            return 32;
        case WeightType::FP16_W:
            return 16;
        case WeightType::INT8_W:
            return 8;
    }
    return 0;
}
template<typename T>
inline WeightType getWeightType()
{
    if (std::is_same<T, float>::value || std::is_same<T, const float>::value) {
        return WeightType::FP32_W;
    }
    else if (std::is_same<T, half>::value || std::is_same<T, const half>::value) {
        return WeightType::FP16_W;
    }
    else if (std::is_same<T, int8_t>::value || std::is_same<T, const int8_t>::value) {
        return WeightType::INT8_W;
    }
    else {
        return WeightType::UNSUPPORTED_W;
    }
}

struct BaseWeight {
//    BaseWeight(std::vector<int> shape_, void* data_, WeightType wtype, void* bias_):
//            shape(shape_), data(data_), type(wtype), bias(bias_){};
    std::vector<int> shape;
    void*   data;
    WeightType type;
    void*   bias;
};
