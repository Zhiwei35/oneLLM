#include <iostream>
#include "src/utils/macro.h"
#include "src/layers/decoder/self_decoder.h"

template<typename T>
void LlamaSelfDecoder::allocForForward(LLaMAAttentionDynParams& params)
{
    // do nothing, no intermedia buffer
}
void LlamaSelfDecoder::free()
{
    // do nothing, no intermedia buffer
}

void LlamaSelfDecoder::forward(TensorMap& input_tensors, const std::vector<LlamaLayerWeight*>& layerWeights, TensorMap& output_tensors, LLaMAAttentionDynParams& dyn_params)
{
    //1. RMSNorm
    Tensor decoder_input = input_tensors["decoder_input"];
    launchRMSNorm(&decoder_input, //in&out, [bs, q_hidden_units]
                  layerWeights[0]->attn_norm_weight,//rmsnorm weights, [q_hidden_units]
                  rmsnorm_eps);
    DeviceSyncAndCheckCudaError();  

    // 2. bias and rope and self attn
    Tensor step = input_tensors["step"];
    Tensor finished = input_tensors["finished"];
    Tensor decoder_output = output_tensors["decoder_output"];
    Tensor all_k_cache = output_tensors["all_k_cache"];
    Tensor all_v_cache = output_tensors["all_v_cache"];
    DataType type_int = getTensorType<int>();
    int layer_id = 0;//TODO: enhance the layer_id update method
    TensorMap self_attn_inputs{
        {"attention_input", decoder_input},
        {"layer_id", Tensor(Device::CPU, type_int, {1}, &layer_id)},
        {"step", step},// a batch shared same step, dim=1 tensor can locate on CPU, no need GPU
        {"finished", finished}
    };
    TensorMap self_attn_outputs{
        {"attention_output", decoder_output},
        {"all_k_cache", all_k_cache},
        {"all_v_cache", all_v_cache}
    };      
    for(int layer_id = 0; layer_id < num_layer; layer_id++) {
        if (layer_id > 0){
            self_attn_inputs["layer_id"] = Tensor(Device::CPU, type_int, {1}, &layer_id);
        }
        //TODO: context_attention.cpp#105, qkv bias should be changed to layerWeights[layer_id].self_attn_weight.qkv.bias
        selfAttn->forward(self_attn_inputs, self_attn_outputs, layerWeights[layer_id]->self_attn_weight, dyn_params);//, selfAttn->GetAttnStaticParams());
        //decoder_output += decoder_input
        launchFusedAddBiasResidualRMSNorm((float*)(decoder_input.data), //in residual, [bs, q hidden_units]
                                          (float*)(decoder_output.data), //in&out, [bs, q hidden_units]
                                          (float*)(layerWeights[layer_id]->self_attn_weight.output.bias), //bias
                                          (float*)(layerWeights[layer_id]->ffn_norm_weight.gamma),//rmsnorm weights, [q hidden_units]
                                          rmsnorm_eps,
                                          dyn_params.batch_size,
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
        launchFusedAddBiasResidualRMSNorm((float*)decoder_input.data, //in, [bs, hidden_units]
                                          (float*)decoder_output.data, //in&out, [bs, hidden_units]
                                          (float*)layerWeights[layer_id]->ffn_weight.down.bias, 
                                          (float*)gamma,//rmsnorm weights, [hidden_units]
                                          rmsnorm_eps,
                                          dyn_params.batch_size,
                                          hidden_units);
        DeviceSyncAndCheckCudaError();
        decoder_input = decoder_output; // for next iter
    }
    // no intermedia buffer to free, so ignore call free
}
