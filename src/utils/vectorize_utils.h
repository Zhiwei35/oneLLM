#include <cuda.h>
#include <cuda_fp16.h>
#define __CUDA_ARCH__ 860
template<typename T_OUT, typename T_IN>
inline __device__ T_OUT scalar_cast_vec(T_IN val)
{
    return val;
}

template<>
inline __device__ half2 scalar_cast_vec<half2, float>(float val)
{
    return __float2half2_rn(val);
}

template<>
inline __device__ float2 scalar_cast_vec<float2, float>(float val)
{
    return make_float2(val, val);
}

template<>
inline __device__ half2 scalar_cast_vec<half2, half>(half val)
{
    return __half2half2(val);
}

// inline __device__ float half_to_float(uint16_t h)
// {
//     float f;
//     asm volatile("cvt.f32.f16 %0, %1;\n" : "=f"(f) : "h"(h));
//     return f;
// }

// inline __device__ uint32_t float2_to_half2(float2 f)
// {
//     union {
//         uint32_t u32;
//         uint16_t u16[2];
//     } tmp;
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
//     asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(tmp.u32) : "f"(f.y), "f"(f.x));
// #else
//     asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[0]) : "f"(f.x));
//     asm volatile("cvt.rn.f16.f32 %0, %1;\n" : "=h"(tmp.u16[1]) : "f"(f.y));
// #endif
//     return tmp.u32;
// }
// inline __device__ float2 half2_to_float2(uint32_t v)
// {
//     uint16_t lo, hi;
//     asm volatile("mov.b32 {%0, %1}, %2;\n" : "=h"(lo), "=h"(hi) : "r"(v));
//     return make_float2(half_to_float(lo), half_to_float(hi));
// }

template<typename T>
struct Vec {
    using Type = T;
    static constexpr int size = 0;
};
template<>
struct Vec<half> {
    using Type = half2; //half2 or uint32_t?
    static constexpr int size = 2;
};
template<>
struct Vec<float> {
    using Type = float4;
    static constexpr int size = 4;
};

struct TwoFloat2{
    float2 x;
    float2 y;
};