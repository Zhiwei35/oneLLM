#include <iostream>
#include "src/kernels/qkv_linear.h"
//TODO: when abstracted weight class, replace T with class
//weight * input
//weight shape = [hidden_units, hidden_units]
//input shape = [hidden_units, seqlen]
// void launchLinearGemm(const float* input,
//                   float* output, 
//                   const int input_2nd_dim, 
//                   const float* weight,
//                   const int hidden_units){}
template<typename T>
void launchLinearGemm(Tensor* input,
                      BaseWeight<T>& weight, 
                      Tensor* output,
                      bool trans_a = false,
                      bool trans_b = false) {
    //TODO: enhance the below 3 obj and setgemmconfig created only once in highest file like ft/bert_example.cc
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    std::cout << "creating stream" << "\n";
    //cudaStreamCreate(&stream);
    // !!!remember to call cublasCreate to create cublas handle!fxxk nvidia, that spent me 1 day to check
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    std::cout << "creating cublaswrapper" << "\n";
    cublasWrapper* cublas_wrapper = 
                        new cublasWrapper(cublas_handle, cublaslt_handle);
                       // , stream);
    cublas_wrapper->setFP32GemmConfig();
    int input_lda = input->shape[0];
    int weight_ldb = input->shape[1];
    // TODO:check 2nd dim of input = 1st dim of weight
    int output_ldc = input_lda;         
    int k = output->shape[1];
    cublasOperation_t transA = trans_a ? CUBLAS_OP_T: CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T: CUBLAS_OP_N;
    std::cout << "calling gemm" << "\n";
    cublas_wrapper->Gemm(transA,
                        transB,
                        input_lda,      //m
                        k,              //n
                        weight_ldb,     //k
                        (T*)input->data,   //A
                        input_lda,      //lda
                        (T*)weight.data,   //B
                        weight_ldb,     //ldb 
                        (T*)output->data,  //C
                        output_ldc,     //ldc   
                        1.0f,
                        0.0f);
}

// We must instancite the template, if not, will report linking issue
template void launchLinearGemm(Tensor* input, BaseWeight<float>& weight, Tensor* output, bool trans_a = false,
                                bool trans_b = false);

template<typename T>
void launchLinearStridedBatchGemm(Tensor* input1,
                                  Tensor* input2,
                                  Tensor* output,
                                  bool trans_a = false,
                                  bool trans_b = false)
{
    cudaStream_t stream;
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    std::cout << "creating stream" << "\n";
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    std::cout << "creating cublaswrapper" << "\n";
    cublasWrapper* cublas_wrapper = 
                        new cublasWrapper(cublas_handle, cublaslt_handle);
                       // , stream);
    cublas_wrapper->setFP32GemmConfig();

    int m = input1->shape[2];
    int k = input1->shape[3];
    int n = input2->shape[2];
    int lda = m;
    int ldb = k;
    int ldc = m;
    int64_t strideA = m * k;
    int64_t strideB = k * n;
    int64_t strideC = m * n;
    // TODO:check 4nd dim of input = 3rd dim of weight
    // TODO:check batchCount of two matrix is equal
    int batchCount = input1->shape[0] * input1->shape[1];

    std::cout << "calling batch gemm" << "\n";
    cublasOperation_t transA = trans_a ? CUBLAS_OP_T: CUBLAS_OP_N;
    cublasOperation_t transB = trans_b ? CUBLAS_OP_T: CUBLAS_OP_N;
    cublas_wrapper->stridedBatchedGemm(transA,
                                       transB,
                                       m,
                                       n,
                                       k,
                                       (T*)input1->data, //A
                                       lda,
                                       strideA,
                                       (T*)input2->data, //B
                                       ldb,
                                       strideB,
                                       (T*)output->data, //C
                                       ldc,
                                       strideC,
                                       batchCount,
                                       1.0f,
                                       0.0f);
}
template void launchLinearStridedBatchGemm<float>(Tensor* input1, Tensor* input2, Tensor* output, bool trans_a = false,
                                        bool trans_b = false);
