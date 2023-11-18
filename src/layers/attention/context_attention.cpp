#include <math.h>
#include "src/layers/attention/context_attention.h"

LLaMAContextAttentionLayer::LLaMAContextAttentionLayer(
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
void LLaMAContextAttentionLayer::allocForForward(LLaMAAttentionDynParams& params) {
    int batch_size = params.batch_size;
    int num_tokens = params.num_tokens;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    const int qkv_head_num = head_num + 2 * kv_head_num;
    //tensor wrapper
    qkv_buf_wo_pad = new Tensor(Device::GPU, type, {num_tokens, qkv_head_num, head_size});
    q_buf_w_pad = new Tensor(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});
    k_buf_w_pad = new Tensor(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size}); //why here isn't max_k_len?cause the q/k/v is got by {bs, q_len, hiddenunits} * {hiddenunits, hiddenunits}
    v_buf_w_pad = new Tensor(Device::GPU, type, {batch_size, kv_head_num, max_q_len, head_size});
    //transpose kv cache
    k_cache_buf = new Tensor(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});// why not kv_head_num？need repeat kv to adapt q head num
    v_cache_buf = new Tensor(Device::GPU, type, {batch_size, head_num, max_k_len, head_size});
    //q*k and softmax
    qk_buf = new Tensor(Device::GPU, type, {batch_size, head_num, max_q_len, max_k_len});
    //qk * v
    qkv_buf_w_pad = new Tensor(Device::GPU, type, {batch_size, head_num, max_q_len, head_size});
    //remove padding
    qkv_buf_wo_pad_1 = new Tensor(Device::GPU, type, {num_tokens, head_num, head_size});
    
    qkv_buf_wo_pad->data = allocator->Malloc(qkv_buf_wo_pad->data, sizeof(T) * num_tokens * qkv_head_num * head_size, false);
    q_buf_w_pad->data = allocator->Malloc(
        q_buf_w_pad->data, sizeof(T) * qkv_head_num * batch_size * max_q_len * head_size, false);
    k_buf_w_pad->data = (float*)q_buf_w_pad->data + head_num * batch_size * max_q_len * head_size;
    v_buf_w_pad->data = (float*)k_buf_w_pad->data + kv_head_num * batch_size * max_q_len * head_size;
    k_cache_buf->data = allocator->Malloc(
        k_cache_buf->data, 2 * sizeof(T) * batch_size * head_num * max_k_len * head_size, false);
    v_cache_buf->data = (float*)k_cache_buf->data + batch_size * head_num * max_k_len * head_size;
    // store qk and inplace store softmax output
    qk_buf->data =
        allocator->Malloc(qk_buf->data, sizeof(T) * batch_size * head_num * max_q_len * max_k_len, false);
    // store qk*v
    qkv_buf_w_pad->data = allocator->Malloc(
        qkv_buf_w_pad->data, sizeof(T) * batch_size * max_q_len * head_num * head_size, false);
    qkv_buf_wo_pad_1->data= allocator->Malloc(qkv_buf_wo_pad->data, sizeof(T) * num_tokens * head_num * head_size, false);

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
    
void LLaMAContextAttentionLayer::free(){
    allocator->deviceFree((void**)(&qkv_buf_wo_pad->data));
    allocator->deviceFree((void**)(&q_buf_w_pad->data));
    allocator->deviceFree((void**)(&k_cache_buf->data));
    allocator->deviceFree((void**)(&v_cache_buf->data));
    allocator->deviceFree((void**)(&qk_buf->data));
    allocator->deviceFree((void**)(&qkv_buf_w_pad->data));
    allocator->deviceFree((void**)(&qkv_buf_wo_pad->data));
}

void LLaMAContextAttentionLayer::forward(TensorMap& inputs, TensorMap& outputs, LLaMAattentionWeights& weights, LLaMAAttentionDynParams& params, LLaMAAttentionStaticParams& static_params)
{   //Can we wrapper the output buf pointer into tensor also?
    //maybe we can create a method to arrange the input tensor and pointer to a struct
    //unifed params order: (input[Tensor], input[Tensor],...,weight[Weight], output[*])
    allocForForward<float>(params);//intermediat buf
    //1.qkv linear
    //[num_tokens, hiddenunits] * [hiddenunits, hiddenunits]
    Tensor attention_input = inputs["attention_input"];
    launchLinearGemm(&attention_input, weights.qkv, qkv_buf_wo_pad);
    //2.qkv bias and rope and padding
    //[num_tokens, hiddenunits]=>{batch_size, q(kv)head_num, max_q_len, head_size}
    Tensor qkv_bias = inputs["qkv_bias"];
    Tensor padding_offset = inputs["padding_offset"];
    Tensor history_length = inputs["history_length"];
    Tensor input_length = inputs["input_length"];
    launchAddFusedQKVBiasTransposeAndRoPE(q_buf_w_pad, k_buf_w_pad, v_buf_w_pad, qkv_buf_wo_pad,
                                        &qkv_bias, &padding_offset, &history_length, &input_length, static_params);
    //3.concat past kv cache
    //max_cache_seq_len = max_seq_len + max_prefix_prompt_length
    //{batch_size, kv_head_num, max_q_len, head_size}=>(num_layer ,batchxbeam ,max_cache_seq_len, hidden_units_};
    Tensor layer_id = inputs["layer_id"];
    //Tensor cur_query_length = inputs["cur_query_length"];
    Tensor all_k_cache = outputs["all_k_cache"];
    Tensor all_v_cache = outputs["all_v_cache"];
    launchAppendKVCache(k_buf_w_pad, v_buf_w_pad, &input_length, &history_length, 
                                &layer_id, &all_k_cache, &all_v_cache);
    //4.MHA/MQA/GQA part, reduce kv cache size to [num_layer, bs, kv head num, max_seq_len, head size]
    //0.kv repeat/broadcast to adapt batchgemm shape requirement([bs, head num, seqlen, head size]) if need
    //[num_layer, bs, kv head num, max_seq_len, head size]=>[bs, q head num, max_k_len, head size]
    Tensor context_length = inputs["context_length"];
    launchTransposeKVCache(&all_k_cache, &all_v_cache, &context_length, 
                                &layer_id, k_cache_buf, v_cache_buf);

    //1.qk [bs,qhead,qlen,headsize]*[bs,qhead,klen,headsize](N*T)=>[bs,head,qlen,klen]
    launchLinearStridedBatchGemm(q_buf_w_pad, k_cache_buf, qk_buf, false, true);

    //2.scale+mask+softmax
    Tensor attention_mask = inputs["attention_mask"];
    launchScaleMaskAndSoftmax(qk_buf, &attention_mask, qk_buf, scale);

    //3.qk*v [bs,head,qlen,klen]=>[bs,head,qlen,headsize]
    launchLinearStridedBatchGemm(qk_buf, v_cache_buf, qkv_buf_w_pad, false, false);

    //4.transpose+reshape([bs,head,seqlen,headsize]=>[bs,seqlen,head,headsize]=>[numtokens,hiddenunits])+remove padding
    launchTransposeOutRemovePadding(qkv_buf_w_pad, &padding_offset, qkv_buf_wo_pad_1);

    // 5.output linear [numtokens,hiddenunits]=>[numtokens,hiddenunits]
    Tensor attention_output = outputs["attention_output"];
    launchLinearGemm(qkv_buf_wo_pad_1, weights.output, &attention_output);

    if (is_free_buffer_after_fwd) {
        this->free();
    }
}
