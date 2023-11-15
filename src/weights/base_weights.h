#include <vector>

enum WeightType
{
    FP32_W,
    FP16_W,
    INT8_W
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
struct BaseWeight {
    // int     input_dims;
    // int     output_dims;
    std::vector<int> shape;
    void*   data;
    WeightType type;
    T*      bias;
};
