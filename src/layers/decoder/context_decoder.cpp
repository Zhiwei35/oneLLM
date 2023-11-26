#include <iostream>
#include "src/layers/decoder/context_decoder.h"
template<typename T>
void LlamaContextDecoder::allocForForward(LLaMAAttentionDynParams& params)
{
    int batch_size = params.batch_size;
    int max_q_len = params.max_q_len;
    int max_k_len = params.max_k_len;
    DataType type = getTensorType<T>(); 
    DataType type_int = getTensorType<int>(); 
    attention_mask = new Tensor(Device::GPU, type, {batch_size, max_q_len, max_k_len});
    padding_offset = new Tensor(Device::GPU, type_int, {batch_size, max_q_len});
    cum_seqlens = new Tensor(Device::GPU, type_int, {batch_size + 1});
    attention_mask->data = allocator->Malloc(attention_mask->data, sizeof(T) * batch_size * max_q_len * max_k_len, false);
    padding_offset->data = allocator->Malloc(padding_offset->data, sizeof(int) * batch_size * max_q_len, false);
    cum_seqlens->data     = allocator->Malloc(cum_seqlens->data, sizeof(int) * (batch_size + 1), false);
   
}
void LlamaContextDecoder::free()
{
    allocator->deviceFree((void**)(&attention_mask->data));
    allocator->deviceFree((void**)(&padding_offset->data));
    allocator->deviceFree((void**)(&cum_seqlens->data));
}
void LlamaContextDecoder::forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams& dyn_params)
{
//     TensorMap context_decoder_inputs{
//         {"decoder_input", Tensor(GPU, type, {attn_dyn_params.num_tokens, hidden_units}, d_attention_input)},
//          {"output_norm_weight", {MEMORY_GPU, dtype, {hidden_units_}, llamaweights->output_norm_weight}},
        // weight {"qkv_bias", Tensor(GPU, type, {head_num * head_size}, d_qkv_bias)},
        // {"padding_offset", Tensor(GPU, type_int, {attn_dyn_params.num_tokens}, d_padding_offset)},
//         {"history_length", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, d_history_len)},
//         {"input_length", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, d_input_len)},
        // {"layer_id", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, d_layer_id)},
//         {"context_length", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, d_ctx_len)},
//         {"attention_mask", Tensor(GPU, type, {attn_dyn_params.batch_size, attn_dyn_params.max_q_len, attn_dyn_params.max_k_len}, d_mask)}
//     };
//     TensorMap context_decoder_outputs{
//         {"decoder_output", Tensor(GPU, type, {attn_dyn_params.num_tokens, q_hidden_units}, d_attention_output)},
//         {"all_k_cache", Tensor(GPU, type,{num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_k_cache)},
//         {"all_v_cache", Tensor(GPU, type, {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_v_cache)}
//     };
    
    allocForForward<float>(dyn_params);
    //1.
    Tensor seq_lens = input_tensors["input_length"];
//    Tensor cum_seqlens = input_tensors["cum_seqlens"];
    Tensor padding_offset = input_tensors["padding_offset"];
//    LLaMAAttentionDynParams dyn_params;
//    dyn_params.batch_size = seq_lens->shape[0];
//    dyn_params.max_q_len = padding_offset->shape[1];
    // shape:
        //seq_lengths:[batch size]
        //output cum_seqlens:[batch size + 1], first ele is 0
        //output padding_offset:[batch size * max q len]
    int h_token_num{};//output 
    launchCalPaddingoffset(h_pinned_token_num_ptr, //pinned host mem alloced in h file
                           &h_token_num, //out
                           (int*)padding_offset.data, //out
                           (int*)cum_seqlens->data, //out
                           (int*)seq_lens.data, // in
                           dyn_params.batch_size,
                           dyn_params.max_q_len);
    //2.
    Tensor attention_mask = input_tensors["attention_mask"];
    Tensor context_length = input_tensors["context_length"];
    //dyn_params.max_k_len = attention_mask->shape[2];
    launchBuildCausalMasks((float*)attention_mask.data, //out
                            (int*)seq_lens.data, //q, input lens, [bs]
                            (int*)context_length.data, //k, context lens, [bs]
                            dyn_params.max_q_len, 
                            dyn_params.max_k_len, 
                            dyn_params.batch_size);
    // 3. RMSnorm
    Tensor decoder_input = input_tensors["decoder_input"];
    dyn_params.num_tokens = decoder_input.shape[0];
    // todo: to enhance the (float*)nullptr
    std::cout << "RMSnorm shape: "<< "\n"
              << "input: "<< decoder_input.shape[0] << "," << decoder_input.shape[1] <<"\n";

    launchFusedAddBiasResidualRMSNorm((float*)nullptr, //in, [num tokens, q_hidden_units]
                                    (float*)decoder_input.data, //in&out, [num tokens, q_hidden_units]
                                    (float*)nullptr,
                                    layerWeights[0]->attn_norm_weight.gamma,//rmsnorm weights, [q_hidden_units]
                                    rmsnorm_eps,
                                    dyn_params.num_tokens,
                                    hidden_units);
    // 4. context attn
    Tensor history_length = input_tensors["history_length"];
    Tensor decoder_output = output_tensors["decoder_output"];
    Tensor all_k_cache = output_tensors["all_k_cache"];
    Tensor all_v_cache = output_tensors["all_v_cache"];
    DataType type_int = getTensorType<int>();
    int layer_id = 0;//TODO: enhance the layer_id update method
    TensorMap ctx_attn_inputs{
        {"attention_input", decoder_input},
        {"padding_offset", padding_offset},
        {"history_length", history_length},
        {"input_length", seq_lens},
        {"context_length", context_length},
        {"attention_mask", attention_mask},
        {"layer_id", Tensor(Device::GPU, type_int, {1}, &layer_id)}
    };
    TensorMap ctx_attn_outputs{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };

    // same buffer between layers, reuse
    for(int layer_id = 0; layer_id < num_layer; layer_id++) {
        if (layer_id > 0){
            ctx_attn_inputs["layer_id"] = Tensor(Device::GPU, type_int, {1}, &layer_id);
        }
        //TODO: context_attention.cpp#105, qkv bias should be changed to layerWeights[layer_id].self_attn_weight.qkv.bias
        ctxAttn->forward(ctx_attn_inputs, ctx_attn_outputs, layerWeights[layer_id]->self_attn_weight, dyn_params, ctxAttn->GetAttnStaticParams());
        //decoder_output += decoder_input
        launchFusedAddBiasResidualRMSNorm((float*)decoder_input.data, //in residual, [num tokens, hidden_units]
                                        (float*)decoder_output.data, //in&out, [num tokens, hidden_units]
                                        (float*)layerWeights[layer_id]->self_attn_weight.output.bias, //bias
                                        (float*)layerWeights[layer_id]->ffn_norm_weight.gamma,//rmsnorm weights, [hidden_units]
                                        rmsnorm_eps,
                                        dyn_params.num_tokens,
                                        hidden_units);
        TensorMap ffn_inputs{
            {"ffn_input", decoder_output}
        };
        TensorMap ffn_outputs{
            {"ffn_output", decoder_output}
        };
        ffn->forward(ffn_inputs, ffn_outputs, layerWeights[layer_id]->ffn_weight, dyn_params);
        auto gamma = layer_id < num_layer - 1 ? layerWeights[layer_id + 1]->attn_norm_weight.gamma :
                                                     (float*)input_tensors["output_norm_weight"].data;//llamaweight->output_norm_weight
        launchFusedAddBiasResidualRMSNorm((float*)decoder_input.data, //in, [num tokens, hidden_units]
                                        (float*)decoder_output.data, //in&out, [num tokens, hidden_units]
                                        (float*)layerWeights[layer_id]->ffn_weight.down.bias, //why add down bias, no bias in down proj
                                        (float*)gamma,//rmsnorm weights, [hidden_units]
                                        rmsnorm_eps,
                                        dyn_params.num_tokens,
                                        hidden_units);
        decoder_input = decoder_output; // for next iter
    }
    if (is_free_buffer_after_forward) {
        free();
    }
}
