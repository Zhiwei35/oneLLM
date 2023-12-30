#pragma once
#include <cuda_runtime.h>
#include <cuda.h>
#include <cublas_v2.h>
#include "src/models/basemodel.h"
#include "src/models/llama/llama.h"
#include "src/utils/macro.h"
#include "src/memory/allocator/cuda_allocator.h"
#include "src/models/llama/llama_params.h"

namespace onellm {
    template<typename T>
    BaseModel *CreateModelWithName(const std::string& model_name) {
        ONELLM_CHECK_WITH_INFO(model_name == "llama", "dont support other models except llama yet!");
        int head_num = 4;// 32;//4;
        int kv_head_num = 2;//32;//2;
        int head_size = 8;//128;//8;
        int inter_size = 12;//11008;//12;
        int num_layers = 2;//32;//2;
        int max_seq_len = 256;
        int vocab_size = 100;//32000;//100;
        int hidden_units = (head_num + 2 * kv_head_num) * head_size;
        int q_hidden_units = head_num * head_size;
        //int step = 0;
        float rmsnorm_eps = 1e-6;
        bool attn_bias = false;
        LLaMAAttentionStaticParams attn_static_params;
        attn_static_params.rotary_embedding_dim = 128;
        attn_static_params.rotary_embedding_base = 10000;
        attn_static_params.max_position_embeddings = 4096;//2048; for llamav1
        attn_static_params.use_dynamic_ntk = false; // for dyn scaling rope
        cublasHandle_t cublas_handle;
        cublasLtHandle_t cublaslt_handle;
        cudaStream_t stream;
        cublasCreate(&cublas_handle);
        cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
        cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
        BaseAllocator* allocator = new CudaAllocator;
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, 0);
        BaseModel *model = new Llama<T>(head_num,
                                        kv_head_num,
                                        head_size,
                                        inter_size,
                                        num_layers,
                                        vocab_size,
                                        attn_static_params,
                                        max_seq_len,
                                        //step,
                                        stream,
                                        cublas_wrapper,
                                        allocator,
                                        true,
                                        &deviceProp);
        return model;
    }

    template<typename T>
    std::unique_ptr<BaseModel> CreateOneLLMModelFromDummy(std::string tokenizer_file){
        BaseModel *model = CreateModelWithName<T>("llama");
        model->loadTokenizer(tokenizer_file);
        model->loadWeightsFromDummy();
        // model->WarmUp();
        return std::unique_ptr<BaseModel> (model);        
    }

    template<typename T>
    std::unique_ptr<BaseModel> CreateOneLLMModelFromFile(std::string model_dir, std::string tokenizer_file){
        BaseModel *model = CreateModelWithName<T>("llama");
        model->loadTokenizer(tokenizer_file);
        model->loadWeights(model_dir);
        // model->WarmUp();
        return std::unique_ptr<BaseModel> (model);        
    }
}
