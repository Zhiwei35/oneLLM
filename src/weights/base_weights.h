#include <vector>

enum WeightType
{
    FP32,
    FP16,
    INT8,
};

inline int getBitNum(WeightType type)
{
    switch (type) {
        case WeightType::FP32:
            return 32;
        case WeightType::FP16:
            return 16;
        case WeightType::INT8:
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
