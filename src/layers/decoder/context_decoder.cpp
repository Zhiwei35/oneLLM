#include <iostream>
#include "src/utils/macro.h"
#include "src/layers/decoder/context_decoder.h"
//TODO: 1.more elegantly call DeviceSyncAndCheckCudaError();
//2.ffn down proj dont have bias, but we cant pass void* to the fusedrmsnorm, here adopt a workaround that allocate 0.0f to bias,  which must be enhanced
//3.line128 update the decoder input of next iter, I think this is right
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
    allocator->deviceFree(attention_mask->data);
    DeviceSyncAndCheckCudaError();
    allocator->deviceFree(padding_offset->data);
    DeviceSyncAndCheckCudaError();
    allocator->deviceFree(cum_seqlens->data);
    DeviceSyncAndCheckCudaError();
}
void LlamaContextDecoder::forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams& dyn_params)
{
    allocForForward<float>(dyn_params);
    //1.
    Tensor seq_lens = input_tensors["input_length"];
    Tensor padding_offset = input_tensors["padding_offset"];
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
    DeviceSyncAndCheckCudaError();
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
    DeviceSyncAndCheckCudaError();
    // 3. RMSnorm
    Tensor decoder_input = input_tensors["decoder_input"];
    dyn_params.num_tokens = decoder_input.shape[0];
    // todo: to enhance the (float*)nullptr
    std::cout << "RMSnorm shape: "<< "\n"
              << "input: "<< decoder_input.shape[0] << "," << decoder_input.shape[1] <<"\n";

    launchRMSNorm(&decoder_input, //in&out, [num tokens, q_hidden_units]
                  layerWeights[0]->attn_norm_weight,//rmsnorm weights, [q_hidden_units]
                  rmsnorm_eps);
    DeviceSyncAndCheckCudaError();
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
        launchFusedAddBiasResidualRMSNorm((float*)(decoder_input.data), //in residual, [num tokens, hidden_units]
                                        (float*)(decoder_output.data), //in&out, [num tokens, hidden_units]
                                        (float*)(layerWeights[layer_id]->self_attn_weight.output.bias), //bias
                                        (float*)(layerWeights[layer_id]->ffn_norm_weight.gamma),//rmsnorm weights, [hidden_units]
                                        rmsnorm_eps,
                                        dyn_params.num_tokens,
                                        hidden_units);
        DeviceSyncAndCheckCudaError();
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
        DeviceSyncAndCheckCudaError();
        decoder_input = decoder_output; // for next iter
    }
    if (is_free_buffer_after_forward) {
        free();
    }
    DeviceSyncAndCheckCudaError();
}
