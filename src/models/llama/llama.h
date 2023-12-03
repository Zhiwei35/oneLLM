#include "src/models/basemodel.h"
#include "src/layers/decoder/context_decoder.h"
#include "src/layers/decoder/self_decoder.h"
#include "src/kernels/input_embedding.h"
#include "src/kernels/qkv_linear.h" //LM Head
#include "src/kernels/beamsearch_topK.h" //topK
#include "src/kernels/topK_sampling.h" //sampling

class Llama: public baseModel{
private:
    const int head_num;
    const int kv_head_num;
    const int head_size;
    const int inter_size;
    const int num_layers;
    const int vocab_size;
    int vocab_size_padded;
    float rmsnorm_eps = 1e-6f;   
    // const int start_id = 0; // from hf modeling_config
    // const int end_id = 2;// from hf modeling_config
    const int hidden_units; 
    const int max_seq_len;
    int output_token_limit = 1000;
    int pad_token_id = 0;// from hf modeling_config 
    int bos_token_id = 1;
    int eos_token_id = 2;

    LlamaWeight* weights;
    LlamaSelfDecoder* self_decoder;
    LlamaContextDecoder* context_decoder;

    const int step;

    T*   context_decoder_input_buf_{};   // CTXDEC
    T*   context_decoder_output_buf_{};  // CTXDEC
    int* context_decoder_ids_buf_{}; //这个倒没见过

    T* decoder_input_buf_{};   // CTXDEC, GENERATE
    T* decoder_output_buf_{};  // CTXDEC, GENERATE

    int* input_ids_buf_{};       // input token ids, CTXDEC
    int* input_length_buf_{};    // input length, CTXDEC, GENERATE
    int* history_length_buf_{};  // history length, CTXDEC
    int* context_length_buf_{};  // history length + input_length, CTXDEC, GENERATE

    // float* logits_buf_{};        // combined logits
    // float* context_logits_buf_{};
    //int* total_padding_count_{};  // GENERATE

    uint64_t* k_cache_ptr_buf_{};
    uint64_t* v_cache_ptr_buf_{};

    // used by sampling
    int*      token_ids_buf_{};   // all token IDs in [S, B], indexed using `step`
    int*      output_ids_buf_{};  // output ids in [B, S]
    int* sequence_lengths_{};     // current sequence length，GENERATE
    //int*      end_ids_buf_{};
    bool*     finished_buf_{};
    // uint32_t* seq_limit_len_{};

    // pinned buffers
    int*       h_input_ids_buf_{};
    int*       h_input_length_buf_{};
    int*       h_history_length_buf_{};
    int*       h_context_length_buf_{};
    int*       h_sequence_lengths_{};
    bool*      h_finished_buf_{};
    uint64_t* h_k_cache_ptr_buf_{};
    uint64_t* h_v_cache_ptr_buf_{};
    // uint32_t*  h_seq_limit_len_{};
public:
    Llama() = default;
    Llama(int head_num,
          int kv_head_num,
          int head_size,
          int inter_size,
          int num_layers,
          int vocab_size,
          const LLaMAAttentionStaticParams&  attn_static_params,
        // int                          max_batch_size,
        // int                          max_context_token_num,
          int max_seq_len,//session_len
          int step,
        // int                          start_id,
        // int                          end_id,
        //for base model
          cudaStream_t stream,
          cublasMMWrapper* cublas_wrapper,
          BaseAllocator* allocator,
          bool is_free_buffer_after_forward,
          cudaDeviceProp* cuda_device_prop):
    BaseModel(stream, cublas_wrapper, allocator, is_free_buffer_after_forward, cuda_device_prop),
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    inter_size(inter_size),
    num_layers(num_layers),
    vocab_size(vocab_size),
    vocab_size_padded(vocab_size),
    step(step),
    hidden_units(head_num * head_size),
    max_seq_len(max_seq_len) {
        self_decoder = new LlamaSelfDecoder(head_num,
                                        kv_head_num,
                                        head_size,
                                        inter_size,
                                        num_layers,
                                        attn_static_params,
                                        rmsnorm_eps,
                                        stream,
                                        cublas_wrapper,
                                        allocator,
                                        is_free_buffer_after_forward);

        context_decoder = new LlamaContextDecoder(head_num,
                                                    kv_head_num,
                                                    head_size,
                                                    inter_size,
                                                    num_layers,
                                                    attn_static_params,
                                                    rmsnorm_eps,
                                                    stream,
                                                    cublas_wrapper,
                                                    allocator,
                                                    is_free_buffer_after_forward);
        allocatePersistBuffer();
    }

    ~Llama() {
        this->free();
    };
    void allocatePersistBuffer(int max_batch_size);
    void allocateBuffer(int batch_size, int session_len);
    void free();
    //weights在common_utils里面已经load好了
    //void loadWeights(std::string file);

    std::string MakeInput(const std::string &history, int round, const std::string &input); // 根据历史信息和当前输入生成prompt

    std::string MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output); // 根据当前轮次回复更新history
    // single request response
    std::string Response(const std::string &input, CallBack PrintRes);

    int forward(TensorMap& inputs, TensorMap& outputs, Weight& weight, DynParams& dparams, StaticParams& sparams);

    void inputEmbedding();

    void InitializeForContextDecoder();

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

    void InitializeForSelfDecoder();

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

    int LMHeadAndTopKSample();
};