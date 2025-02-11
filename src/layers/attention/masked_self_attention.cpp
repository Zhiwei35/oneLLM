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
    allocator->deviceFree(qkv_buf->data);
    DeviceSyncAndCheckCudaError();
    allocator->deviceFree(mha_output->data);
    DeviceSyncAndCheckCudaError();
}

void LLaMASelfAttentionLayer::forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights& weights, LLaMAAttentionDynParams& params)
{   
    //maybe we can create a method to arrange the input tensor and pointer to a struct
    //unifed params order: (input[Tensor], input[Tensor],...,weight[Weight], output[*])
    allocForForward<float>(params);//intermediat buf
    //1. qkv linear
    //[bs,1,q_hidden_units] * [q_hidden_units, hidden_units] = [bs,1,hidden_units]
    Tensor attention_input = inputs["attention_input"];
    launchLinearGemm(&attention_input, weights.qkv, qkv_buf);

    //2. biasrope + masked mha
    //目前和FT lmdeploy相比少了total_padding_len(用在rope，timestep-=padlen（合理，不用对pad求rope），在llamabatch::initializeGenerate函数里面得到) sequence_lengths（每个句子所有轮的总长度，用在求tlength dynamic_ntk下的rotary_embedding_base）
    Tensor attention_output = outputs["attention_output"];
    //[step, bs, kv head num, head size],貌似少了一个layerid这样一个shape，后面看看添到哪维
    Tensor key_cache       = outputs["all_k_cache"]; // prepared in llamacachemgr and llamabatch::initialize
    Tensor value_cache     = outputs["all_v_cache"];
    Tensor finished = inputs["finished"];
    // Tensor total_padding_len = inputs["total_padding_len"]; //[bs], for rope
    Tensor step = inputs["step"];//[1]
    Tensor layer_id = inputs["layer_id"];//[1]

    launchDecoderMaskedMHA(qkv_buf, weights.qkv, &key_cache, &value_cache, &finished, &step, mha_output, attn_static_params);
    DeviceSyncAndCheckCudaError();

    launchLinearGemm(mha_output, weights.output, &attention_output);
    if (is_free_buffer_after_fwd) {
        this->free();
        DeviceSyncAndCheckCudaError();
    }
    //seqlen将在sampling更新
    //Tensor sequence_lengths = inputs["sequence_lengths"]; //[bs] length_per_sample, 贯穿全kernel，to get tlength, to decide kv cache seqlen, that is kv cache's step/seqlen
}
