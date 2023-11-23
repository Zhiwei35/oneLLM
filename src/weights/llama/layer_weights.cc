#include "src/weights/llama/layer_weights.h"
LlamaLayerWeight::LlamaLayerWeight(int     head_num,
                                    int     kv_head_num,
                                    int     head_size,
                                    int     inter_size,
                                    WeightType weight_type,
                                    //int        group_size,
                                    bool       attn_bias):
    head_num(head_num),
    kv_head_num(kv_head_num),
    head_size(head_size),
    hidden_units(head_num * head_size),
    inter_size(inter_size),
    weight_type(weight_type),
    attn_bias(attn_bias)
{
    // init weights structure and cudamalloc for weights
    GPUMalloc(&attn_norm_weight.gamma, hidden_units);
    GPUMalloc(&ffn_norm_weight.gamma, hidden_units);
    self_attn_weight.qkv.type = weight_type;
    self_attn_weight.qkv.shape = {(head_num + 2 * kv_head_num), head_size};
    GPUMalloc(&self_attn_weight.qkv.data, (head_num + 2 * kv_head_num)* head_size);
    self_attn_weight.output.type = weight_type;
    self_attn_weight.output.shape = {head_num, head_size};
    GPUMalloc(&self_attn_weight.output.data, head_num * head_size);
    if (attn_bias) {
        GPUMalloc(&self_attn_weight.qkv.bias, (head_num + 2 * kv_head_num)* head_size);
        GPUMalloc(&self_attn_weight.output.bias, head_num * head_size);
    }

    ffn_weight.gate.type = weight_type;
    ffn_weight.up.type = weight_type;
    ffn_weight.down.type = weight_type;
    ffn_weight.gate.shape = {hidden_units, inter_size};
    ffn_weight.up.shape = {hidden_units, inter_size};
    ffn_weight.down.shape = {inter_size, hidden_units};
    GPUMalloc(&self_attn_weight.gate.data, hidden_units * inter_size);
    GPUMalloc(&self_attn_weight.up.data, hidden_units * inter_size);
    GPUMalloc(&self_attn_weight.down.data, hidden_units * inter_size);
}
//model file type用来控制loadweightfrombin的第二个模板类型参数T_IN,如果T_IN和第一个模板参数不一致，需要将T_IN的weight使用ptx cast转换为T
LlamaLayerWeight::loadWeights(std::string weight_path, WeightType weight_type)
{
    loadWeightFromBin<float, float>(attn_norm_weight.gamma, {hidden_units}, weight_path + ".attention_norm.weight");
    loadWeightFromBin<float, float>(ffn_norm_weight.gamma, {hidden_units}, weight_path + ".ffn_norm.weight");

    loadWeightFromBin<float, float>(self_attn_weight.qkv.data, {(head_num + 2 * kv_head_num), head_size}, weight_path + ".attention.w_qkv.weight");
    loadWeightFromBin<float, float>(self_attn_weight.output.data, {head_num, head_size}, weight_path + ".attention.wo.weight");
    loadWeightFromBin<float, float>(ffn_weight.gate.data, {hidden_units, inter_size}, weight_path + ".feed_forward.w1.weight");
    loadWeightFromBin<float, float>(ffn_weight.up.data, {hidden_units, inter_size}, weight_path + ".feed_forward.w3.weight");
    loadWeightFromBin<float, float>(ffn_weight.down.data, {inter_size, hidden_units}, weight_path + ".feed_forward.w2.weight");
    if (attn_bias) {//TODO
        loadWeightFromBin<float, float>(self_attn_weight.qkv.bias, {(head_num + 2 * kv_head_num), head_size}, weight_path + ".attention.w_qkv.bias");
        loadWeightFromBin<float, float>(self_attn_weight.output.bias, {head_num, head_size}, weight_path + ".attention.wo.bias");
    }   
}

void freeWeights(BaseWeight& weights)
{
    cudaFree(weights.data);
    if(bias != nullptr) {
        cudaFree(weights.bias);
    }

    weights.data = nullptr;
    weights.bias = nullptr;
}

LlamaLayerWeight::~LlamaLayerWeight()
{
    // free norm
    cudaFree(attn_norm_weight.gamma);
    cudaFree(ffn_norm_weight.gamma);
    //free weights, including data and bias
    freeWeights(self_attn_weight.qkv);
    freeWeights(self_attn_weight.output);
    freeWeights(ffn_weight.gate);
    freeWeights(ffn_weight.up);
    freeWeights(ffn_weight.down);
}

