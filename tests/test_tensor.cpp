#include <iostream>
#include <unordered_map>
#include <vector>
#include <gtest/gtest.h>
#include "src/utils/tensor.h"

void test1(){
    bool*  v1 = new bool(true);
    float* v2 = new float[6]{1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f};
    Tensor t1 = Tensor{CPU, BOOL, {1}, v1};
    Tensor t2 = Tensor{CPU, FP32, {3, 2}, v2};

    TensorMap map({{"t1", t1}, {"t2", t2}});
    if(map.isExist("t1")) {
        std::cout << "t1 exist, expected true" << "\n";
    }
    if(map.isExist("t2")){
        std::cout << "t2 exist, expected true" << "\n";
    }
    if(map.isExist("t3")){
        std::cout << "t3 exist, expected true" << "\n";
    }

    delete v1;
    delete[] v2;
}
void test2()
{
    int*   v1 = new int[4]{1, 10, 20, 30};
    float* v2 = new float[2]{1.0f, 2.0f};
    Tensor t1 = Tensor(MEMORY_CPU, TYPE_INT32, {4}, v1);
    Tensor t2 = Tensor(MEMORY_CPU, TYPE_INT32, {2}, v2);

    TensorMap map({{"t1", t1}});
    if(map.size() == 1){
        std::cout << "map.size() = 1" << "\n";
    }
    if(map.isExist("t1")){
        std::cout << "t1 existed in map" << "\n";
    }
    if(map.at("t1") == t1){
        std::cout << "t1 tensor in map is ture" << "\n";
    }
    if(map.isExist("t2")){
        std::cout << "t2 existed in map" << "\n";
    }
}

int main() {
    test1();
    test2();
    return 0;
}