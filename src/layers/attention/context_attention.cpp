#include "src/layers/attention/context_attention.h"
#include "src/kernels/qkv_linear.h"
#include "src/utils/tensor.h"

LLaMAContextAttentionLayer<T>::LLaMAContextAttentionLayer(
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
    is_free_buffer_after_fwd(is_free_buffer_after_fwd) {
        // TODO: check kv_head_num is divided by haed_num
        q_head_per_kv = head_num / kv_head_num;
    }
template<typename T>
LLaMAContextAttentionLayer<T>::allocForForward(LLaMAAttentionDynParams params) {
    int batch_size = params.batch_size;
    int num_tokens = params.num_tokens;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    int num_layers = params.num_layers;  
    DataType type = getTensorType<T>(); 
    const int qkv_head_num = head_num + 2 * kv_head_num;
    //tensor wrapper
    qkv_buf_wo_pad = new Tensor(Device::GPU, type, {num_tokens, qkv_head_num, head_size});
    q_buf_w_pad = new Tensor(Device::GPU, type, {batch_size, max_q_len, head_num, head_size});
    k_buf_w_pad = new Tensor(Device::GPU, type, {batch_size, max_q_len, kv_head_num, head_size}); //why here isn't max_k_len, maybe max_k_len is the max k length across all epochs, max q len is the max lenght of cur epoch
    v_buf_w_pad = new Tensor(Device::GPU, type, {batch_size, max_q_len, kv_head_num, head_size});
    k_cache_buf = new Tensor(Device::GPU, type, {num_layers, batch_size, max_k_len, head_num, head_size});// why not kv_head_num
    v_cache_buf = new Tensor(Device::GPU, type, {num_layers, batch_size, max_k_len, head_num, head_size});
    qk_buf = new Tensor(Device::GPU, type, {batch_size, head_num, max_q_len, max_k_len});
    qkv_buf_w_pad = new Tensor(Device::GPU, type, {batch_size, max_q_len, head_num, head_size});
    qkv_buf_wo_pad_1 = new Tensor(Device::GPU, type, {num_tokens, head_num, head_size});
    
    qkv_buf_wo_pad->data = allocator->Malloc(qkv_buf_wo_pad->data, sizeof(T) * num_tokens * qkv_head_num * head_size);
    q_buf_w_pad->data = allocator->Malloc(
        q_buf_w_pad->data, sizeof(T) * qkv_head_num * batch_size * max_q_len * head_size);
    k_buf_w_pad->data = q_buf_w_pad->data + head_num * batch_size * max_q_len * head_size;
    v_buf_w_pad->data = k_buf_w_pad->data + kv_head_num * batch_size * max_q_len * head_size;
    k_cache_buf->data = allocator->Malloc(
        k_cache_buf->data, 2 * sizeof(T) * num_layers * batch_size * head_num * max_k_len * head_size);
    v_cache_buf->data = k_cache_buf->data + num_layers * batch_size * head_num * max_k_len * head_size;
    // store qk and inplace store softmax output
    qk_buf->data =
        allocator->Malloc(qk_buf->data, sizeof(T) * batch_size * head_num * max_q_len * max_k_len);
    // store qk*v
    qkv_buf_w_pad->data = allocator->Malloc(
        qkv_buf_w_pad->data, sizeof(T) * batch_size * max_q_len * head_num * head_size);
    qkv_buf_wo_pad_1->data= allocator->Malloc(qkv_buf_wo_pad->data, sizeof(T) * num_tokens * head_num * head_size);

    // directly pointer
    // qkv_buf_wo_pad = allocator->Malloc(qkv_buf_wo_pad, sizeof(T) * num_tokens * qkv_head_num * head_size);
    // q_buf_w_pad = allocator->Malloc(
    //     q_buf_w_pad, sizeof(T) * qkv_head_num * batch_size * max_q_len * head_size);
    // k_buf_w_pad = q_buf_w_pad + head_num * batch_size * max_q_len * head_size;
    // v_buf_w_pad = k_buf_w_pad + kv_head_num * batch_size * max_q_len * head_size;
    // k_cache_buf = allocator->Malloc(
    //     k_cache_buf, 2 * sizeof(T) * num_layers * batch_size * head_num * max_k_len * head_size);
    // v_cache_buf = k_cache_buf + num_layers * batch_size * head_num * max_k_len * head_size;
    // // store qk and inplace store softmax output
    // qk_buf =
    //     allocator->Malloc(qk_buf, sizeof(T) * batch_size * head_num * max_q_len * max_k_len);
    // // store qk*v
    // qkv_buf_w_pad = allocator->Malloc(
    //     qkv_buf_w_pad, sizeof(T) * batch_size * max_q_len * head_num * head_size);
    // qkv_buf_wo_pad= allocator->Malloc(qkv_buf_wo_pad, sizeof(T) * num_tokens * head_num * head_size);

}
    
template<typename T>
void LLaMAContextAttentionLayer<T>::freeBuffer(){
    allocator->free((void**)(&qkv_buf_wo_pad));
    allocator->free((void**)(&q_buf_w_pad));
    allocator->free((void**)(&k_cache_buf));
    allocator->free((void**)(&v_cache_buf));
    allocator->free((void**)(&qk_buf));
    allocator->free((void**)(&qkv_buf_w_pad));
    allocator->free((void**)(&qkv_buf_wo_pad));
}

template<typename T>
void LLaMAContextAttentionLayer<T>::forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights<T>& weights)
{   //Can we wrapper the output buf pointer into tensor also?
    //maybe we can create a method to arrange the input tensor and pointer to a struct
    //unifed params order: (input[Tensor], input[Tensor],...,weight[Weight], output[*])
    LLaMAAttentionDynParams params;
    allocForForward(params);//intermediat buf
    //1.qkv linear
    // [hiddenunits, hiddenunits] * [hiddenunits, num_tokens]
    // or [num_tokens, hiddenunits] * [hiddenunits, hiddenunits]
    Tensor attention_input = inputs["attention_input"];
    launchLinearGemm(attention_input, weights->qkv, qkv_buf_wo_pad);
    //2.qkv bias and rope
    Tensor attention_input = inputs["qkv_bias"];
    Tensor padding_offset = inputs["padding_offset"];
    Tensor history_length = inputs["history_length"];
    Tensor input_length = inputs["input_length"];
    launchAddFusedQKVBiasTransposeAndRoPE(q_buf_w_pad, k_buf_w_pad, v_buf_w_pad, qkv_buf_wo_pad
                                        attention_input, padding_offset, history_length, input_length);
    //3.concat past kv cache
    Tensor layer_id = inputs["layer_id"];
    Tensor cur_query_length = inputs["cur_query_length"];
    launchAppendKVCache(k_cache_buf, v_cache_buf, layer_id, k_buf_w_pad, v_buf_w_pad, 
                            cur_query_length, history_length);
    //4.MHA/MQA/GQA part
    //0.kv transpose to adapt batchgemm shape requirement([bs, head num, seqlen, head size]) if need
    Tensor context_length = inputs["context_length"];
    Tensor all_k_cache = outputs["all_k_cache"];
    Tensor all_v_cache = outputs["all_v_cache"];
    launchTransposeKVCache(k_cache_buf, v_cache_buf, layer_id, q_head_per_kv, context_length, all_k_cache, all_v_cache);

    //1.qk
    launchLinearStridedBatchGemm(q_buf_w_pad, k_cache_buf, qk_buf);

    //2.scale+mask+softmax
    Tensor attention_mask = inputs["attention_mask"];
    launchMaskedSoftmax(qk_buf, attention_mask, qk_buf);

    //3.qk*v
    launchLinearStridedBatchGemm(qk_buf, v_cache_buf, qkv_buf_w_pad);

    //4.transpose+reshape([bs,head,seqlen,headsize]=>[bs,seqlen,head,headsize]=>[bs,seqlen,hiddenunits])+remove padding
    launchTransposeOutRemovePadding(qkv_buf_w_pad, qkv_buf_wo_pad);

    // 5.output linear
    Tensor attention_output = outputs["attention_output"];
    launchLinearGemm(qkv_buf_wo_pad, weights->output, attention_output);

    if (is_free_buffer_after_fwd) {
        freeBuffer();
    }
}