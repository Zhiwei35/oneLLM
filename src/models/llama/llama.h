#include "src/models/basemodel.h"
#include "src/layers/decoder/context_decoder.h"
#include "src/layers/decoder/self_decoder.h"
#include "src/kernels/qkv_linear.h" //LM Head
#include "src/kernels/beamsearch_topK.h" //topK
#include "src/kernels/topK_sampling.h" //sampling

class Llama {
private:
    const int head_num;
    const int head_size;
    const int inter_size;
    const int num_layer;
    const int vocab_size;
    int vocab_size_padded;
    float rmsnorm_eps = 1e-6f;   
    const int start_id;
    const int end_id;
    const int hidden_units; 

    LlamaWeight* weights;
    LlamaSelfDecoder* decoder;
    LlamaContextDecoder* context_decoder;

    const int step;
public:
    Llama() = default;
    Llama(int                       head_num,
        int                       kv_head_num,
        int                       head_size,
        int                       inter_size,
        int                       num_layer,
        int                       vocab_size,
        const LLaMAAttentionStaticParams&  attn_params,
        float                        norm_eps,
        int                          max_batch_size,
        int                          max_context_token_num,
        int                          max_seq_len,//session_len
        int                          step,
        int                          start_id,
        int                          end_id,
        //for base model
        cudaStream_t                 stream,
        cublasMMWrapper*             cublas_wrapper,
        BaseAllocator*               allocator,
        bool                         is_free_buffer_after_forward,
        cudaDeviceProp*              cuda_device_prop);

    ~Llama();
    
    void loadWeights(std::string file);

    std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

    std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前轮次回复更新history

    std::string Response(const std::string &input, CallBack PrintRes);

    int forward(TensorMap& inputs, TensorMap& outputs, Weight& weight, DynParams& dparams, StaticParams& sparams);

    //placehoder, need to modify
    void contextDecode(T*         deocder_output,
                       uintptr_t* k_cache_ptr,
                       uintptr_t* v_cache_ptr,
                       T*         context_decoder_input_buf,
                       T*         context_decoder_output_buf,
                       const int* input_ids,
                       const int* input_length,
                       const int* history_length,
                       const int* context_length,
                       size_t     token_num,
                       size_t     max_input_len,
                       size_t     max_context_len,
                       size_t     session_len,
                       size_t     batch_size);

    void selfDecoder(T*         decoder_output,
                    uintptr_t* k_cache_ptr,
                    uintptr_t* v_cache_ptr,
                    T*         decoder_input,
                    const int* sequence_length,
                    const int* total_padding_count,
                    bool*      finished,
                    int        step,
                    int        ite,
                    size_t     session_len,
                    size_t     batch_size);
};