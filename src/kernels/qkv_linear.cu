#include <iostream>
#include "src/kernels/qkv_linear.h"
//TODO: when abstracted weight class, replace T with class
//weight * input
//weight shape = [hidden_units, hidden_units]
//input shape = [hidden_units, seqlen]
//template<typename T>
void launchLinear(const float* input,
                  float* output, 
                  const int input_2nd_dim, 
                  const float* weight,
                  const int hidden_units) {
    //TODO: enhance the below 3 obj and setgemmconfig created only once in highest file like ft/bert_example.cc
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    std::cout << "before create stream" << "\n";
    //cudaStreamCreate(&stream);
    // !!!remember to call cublasCreate to create cublas handle!fxxk nvidia, that spent me 1 day to check
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    std::cout << "before create cublaswrapper" << "\n";
    cublasWrapper* cublas_wrapper = 
                        new cublasWrapper(cublas_handle, cublaslt_handle);
                       // , stream);
    cublas_wrapper->setFP32GemmConfig();
    std::cout << "before call gemm" << "\n";
    cublas_wrapper->Gemm(CUBLAS_OP_N,
                        CUBLAS_OP_N,
                        hidden_units, //weight.output_dims,         //m
                        input_2nd_dim,                 //n
                        hidden_units, //weight.input_dims,          //k
                        weight,//(const T*)weight.kernel,    //A
                        hidden_units, //weight.output_dims,        //lda
                        input,                      //B
                        hidden_units, //weight.input_dims, 
                        output,                     //C
                        hidden_units, //weight.output_dims,    
                        1.0f,
                        0.0f);
}

// We must instancite the template, if not, will report linking issue
//template void launchLinear(const float* input, float* output, const int input_2nd_dim, const float* weight, const int hidden_units);
