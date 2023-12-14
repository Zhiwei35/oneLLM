#include "cublas_wrapper.h"
#include <iostream>
#define CUBLAS_WORKSPACE_SIZE 32*1024*1024
//waiting for final testing after developing allocator
//can test first with mannual cudaMalloc
cublasWrapper::cublasWrapper(cublasHandle_t cublas_handle,
                                 cublasLtHandle_t cublaslt_handle):
//                                 cudaStream_t stream):
                                 //BaseAllocator* allocator):
    cublas_handle_(cublas_handle),
    cublaslt_handle_(cublaslt_handle)
//    stream_(stream)
    //allocator_(allocator) // cublas workspace will only be used in cublaslt API and algo search
{
    // if (allocator_ != nullptr) {
    //     cublas_workspace_ = allocator_->reMalloc(cublas_workspace_, CUBLAS_WORKSPACE_SIZE, false);
    // }
}

cublasWrapper::~cublasWrapper()
{
//    if (allocator_) {
//        allocator_->free((void**)(&cublas_workspace_));
//        allocator_ = nullptr;
//    }
}
// invoked in model example main function after initialize cublas wrapper
void cublasWrapper::setFP32GemmConfig()
{
    Atype_       = CUDA_R_32F;
    Btype_       = CUDA_R_32F;
    Ctype_       = CUDA_R_32F;
    computeType_ = CUBLAS_COMPUTE_32F;
}

void cublasWrapper::setFP16GemmConfig()
{
    Atype_       = CUDA_R_16F;
    Btype_       = CUDA_R_16F;
    Ctype_       = CUDA_R_16F;
    computeType_ = CUBLAS_COMPUTE_16F; // cuda version > 11.0的写法, <的时候是CUDA_R_16F的写法
}

//fp32 gemm and fp16 gemm
void cublasWrapper::Gemm(cublasOperation_t transa,
                           cublasOperation_t transb,
                           const int         m,
                           const int         n,
                           const int         k,
                           const void*       A,
                           const int         lda,
                           const void*       B,
                           const int         ldb,
                           void*             C,
                           const int         ldc,
                           float             f_alpha = 1.0f,
                           float             f_beta = 0.0f)
{
    half h_alpha = (half)(f_alpha);
    half h_beta  = (half)(f_beta);
    int is_fp16_computeType = computeType_ == CUBLAS_COMPUTE_16F ? 1 : 0; //之前是CUDA_R_16F
    const void* alpha = is_fp16_computeType ? reinterpret_cast<void*>(&(h_alpha)) : reinterpret_cast<void*>(&f_alpha);
    const void* beta  = is_fp16_computeType ? reinterpret_cast<void*>(&(h_beta)) : reinterpret_cast<void*>(&f_beta);
    // const void* alpha = reinterpret_cast<void*>(&f_alpha);
    // const void* beta = reinterpret_cast<void*>(&f_beta);
    CHECK_CUBLAS(cublasGemmEx(cublas_handle_,
                            transa,
                            transb,
                            m,
                            n,
                            k,
                            alpha,
                            A,
                            Atype_,
                            lda,
                            B,
                            Btype_,
                            ldb,
                            beta,
                            C,
                            Ctype_,
                            ldc,
                            computeType_,
                            CUBLAS_GEMM_DEFAULT));
}

void cublasWrapper::stridedBatchedGemm(cublasOperation_t transa,
                                        cublasOperation_t transb,
                                        const int         m,
                                        const int         n,
                                        const int         k,
                                        const void*       A,
                                        const int         lda,
                                        const int64_t     strideA,
                                        const void*       B,
                                        const int         ldb,
                                        const int64_t     strideB,
                                        void*             C,
                                        const int         ldc,
                                        const int64_t     strideC,
                                        const int         batchCount,
                                        float       f_alpha = 1.0f,
                                        float       f_beta  = 0.0f)
{
    int is_fp16_computeType = computeType_ == CUDA_R_16F ? 1 : 0;
    const void* alpha =
       is_fp16_computeType ? reinterpret_cast<void*>(&(f_alpha)) : reinterpret_cast<const void*>(&f_alpha);
    const void* beta = is_fp16_computeType ? reinterpret_cast<void*>(&(f_beta)) : reinterpret_cast<const void*>(&f_beta);
    // const void* alpha = reinterpret_cast<void*>(&f_alpha);
    // const void* beta = reinterpret_cast<void*>(&f_beta);
    // std::cout << "m,n,k=" << m << "," << n << "," << k << "\n"
    //           << "lda,ldb,ldc=" << lda << "," << ldb << "," << ldc << "\n"
    //           << "strideA,strideB,strideC=" << strideA << "," << strideB << "," << strideB << "\n"
    //           << "batchcount=" << batchCount << "\n";
    CHECK_CUBLAS(cublasGemmStridedBatchedEx(cublas_handle_,
                                            transa,
                                            transb,
                                            m,
                                            n,
                                            k,
                                            alpha,
                                            A,
                                            Atype_,
                                            lda,
                                            strideA,
                                            B,
                                            Btype_,
                                            ldb,
                                            strideB,
                                            beta,
                                            C,
                                            Ctype_,
                                            ldc,
                                            strideC,
                                            batchCount,
                                            computeType_,
                                            CUBLAS_GEMM_DEFAULT));
}
