#include <math.h>
#include "src/layers/attention/masked_self_attention.h"

LLaMASelfAttentionLayer::LLaMASelfAttentionLayer(
                               int head_num,
                               int kv_head_num,
                               int head_size,
                               LLaMAAttentionStaticParams attn_params,
                               cudaStream_t stream,
                               cublasWrapper* cublas_wrapper,
                               BaseAllocator* allocator,
                               bool is_free_buffer_after_fwd):
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    stream(stream),
    cublas_wrapper(cublas_wrapper),
    allocator(allocator), //cudaAllocator
    hidden_units(head_num * head_size),
    attn_static_params(attn_params),
    is_free_buffer_after_fwd(is_free_buffer_after_fwd),
        // TODO: check kv_head_num is divided by haed_num
    q_head_per_kv(head_num / kv_head_num),
    scale(float(1 / sqrt(head_size))){}

template<typename T>
void LLaMASelfAttentionLayer::allocForForward(LLaMAAttentionDynParams& params) {
    int batch_size = params.batch_size;
    int num_tokens = params.num_tokens;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    const int qkv_head_num = head_num + 2 * kv_head_num;
    //tensor wrapper
    // 当前step的q k v的shape里面step或seqlen都是1，之前step的kv在做gemv的时候直接从kv cache拿
    qkv_buf = new Tensor(Device::GPU, type, {batch_size, qkv_head_num, head_size}); 
    mha_output = new Tensor(Device::GPU, type, {batch_size, hidden_units});
    
    qkv_buf->data = allocator->Malloc(qkv_buf->data, sizeof(T) * batch_size * qkv_head_num * head_size, false);
    mha_output->data = allocator->Malloc(
        mha_output->data, sizeof(T) * batch_size * hidden_units, false);
}

void LLaMASelfAttentionLayer::free(){
    allocator->deviceFree((void**)(&qkv_buf->data));
    allocator->deviceFree((void**)(&mha_output->data));
}

void LLaMASelfAttentionLayer::forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights& weights, LLaMAAttentionDynParams& params, LLaMAAttentionStaticParams& static_params)
{   
    //maybe we can create a method to arrange the input tensor and pointer to a struct
    //unifed params order: (input[Tensor], input[Tensor],...,weight[Weight], output[*])
    allocForForward<float>(params);//intermediat buf
    //1. qkv linear
    Tensor attention_input = inputs["attention_input"];
    launchLinearGemm(&attention_input, weights.qkv, qkv_buf);

    //2. masked mha
    Tensor attention_output = outputs["attention_output"];
    Tensor key_cache       = outputs["key_cache"];
    Tensor value_cache     = outputs["value_cache"];
    Tensor finished = inputs["finished"];
    Tensor sequence_lengths = inputs["sequence_lengths"]; // length_per_sample, 贯穿全kernel，to get tlength, to decide kv cache seqlen, that is kv cache's step/seqlen
    Tensor total_padding_len = inputs["total_padding_len"]; //[bs], for rope
    Tensor step = inputs["step"];//[1]
    Tensor layer_id = inputs["layer_id"];//[1]
    launchDecoderMaskedMHA(qkv_buf, &key_cache, &value_cache, &finished, &step, mha_output);

    launchLinearGemm(mha_output, weights.output, &attention_output);
    if (is_free_buffer_after_fwd) {
        this->free();
    }
}