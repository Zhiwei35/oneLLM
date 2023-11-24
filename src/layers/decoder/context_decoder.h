#include "src/kernels/build_casual_mask.h"
#include "src/kernels/cal_paddingoffset.h"
#include "src/kernels/fused_addresidual_norm.h"
#include "src/layers/attention/context_attention.h"
#include "src/layers/ffn/ffn.h"
#include "src/weights/llama/llama_weights.h"
#include "src/utils/tensor.h"

//layer weights is ready at the beginning                                                                                                                                             by loadweights in onellm.cpp, outside of the decoder
class LlamaContextDecoder{
private:
    int head_num;
    int kv_head_num;
    int head_size;
    int inter_size;
    int num_layer;
    int hidden_units;
    float rmsnorm_eps; 
    Tensor* attention_mask;
    Tensor* padding_offset;
    Tensor* cum_seqlens;
    int* h_pinned_token_num_ptr;
    cudaStream_t stream;
    cublasWrapper* cublas_wrapper;
    BaseAllocator* allocator;
    bool is_free_buffer_after_forward;


    LLaMAContextAttentionLayer* ctxAttn;
    LLaMAFFNLayer* ffn;
    DataType data_type;
    // all layers' weight没必要作为成员，只需从forward里面传进去就完了，初始化是完成在model.loadweights阶段
    //const std::vector<LlamaLayerWeight*> weights;
    // struct Session {//除了dyn_params，没看出sess内的其它成员为什么要作为该类成员，都是外部作为tensor传进来的
    //LLaMAAttentionDynParams& dyn_params;
    // Tensor* k_cache;
    // Tensor* v_cache;
    // int* input_length;
    // int* history_length;
    // int* context_length;
    // };
public:
    LlamaContextDecoder(int                      head_num,
                        int                      kv_head_num,
                        int                      head_size,
                        int                      inter_size,
                        int                      num_layer,
                        const LLaMAAttentionStaticParams& attn_params,
                        float                       rmsnorm_eps,
                        cudaStream_t                stream,
                        cublasWrapper*            cublas_wrapper,
                        BaseAllocator*                 allocator,
                        bool                     is_free_buffer_after_forward):
        head_num(head_num),
        head_size(head_size),
        inter_size(inter_size),
        hidden_units(head_num * head_size),
        num_layer(num_layer),
        rmsnorm_eps(rmsnorm_eps),
        data_type(getTensorType<float>()),
        stream(stream),
        cublas_wrapper(cublas_wrapper),
        allocator(allocator),
        is_free_buffer_after_forward(is_free_buffer_after_forward){
            h_pinned_token_num_ptr = (int*)allocator->Malloc(h_pinned_token_num_ptr, sizeof(size_t), true);
            ctxAttn = new LLaMAContextAttentionLayer(head_num,
                                                        kv_head_num,
                                                        head_size,
                                                        attn_params,
                                                        stream,
                                                        cublas_wrapper,
                                                        allocator,
                                                        is_free_buffer_after_forward);

            ffn = new LLaMAFFNLayer(head_num,
                                    head_size,
                                    inter_size,
                                    stream,
                                    cublas_wrapper,
                                    allocator,
                                    is_free_buffer_after_forward);
        };
    template<typename T>
    void allocForForward(LLaMAAttentionDynParams& dyn_params);
    void free();
    void forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams& dyn_params);
};
