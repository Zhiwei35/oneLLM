#include "src/weights/llama/layer_weights.h"
template<typename T>
LlamaLayerWeight<T>::LlamaLayerWeight(int     head_num,
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
    self_attn_weight.qkv.shape = {hidden_units, (head_num + 2 * kv_head_num) * head_size};
    GPUMalloc(&self_attn_weight.qkv.data, hidden_units * (head_num + 2 * kv_head_num) * head_size);
    self_attn_weight.output.type = weight_type;
    self_attn_weight.output.shape = {hidden_units, hidden_units};
    GPUMalloc(&self_attn_weight.output.data, hidden_units * hidden_units);
    if (attn_bias) {
        GPUMalloc(&self_attn_weight.qkv.bias, (head_num + 2 * kv_head_num) * head_size);
        GPUMalloc(&self_attn_weight.output.bias, hidden_units);
    }

    ffn_weight.gate.type = weight_type;
    ffn_weight.up.type = weight_type;
    ffn_weight.down.type = weight_type;
    ffn_weight.gate.shape = {hidden_units, inter_size};
    ffn_weight.up.shape = {hidden_units, inter_size};
    ffn_weight.down.shape = {inter_size, hidden_units};
    GPUMalloc(&ffn_weight.gate.data, hidden_units * inter_size);
    GPUMalloc(&ffn_weight.up.data, hidden_units * inter_size);
    GPUMalloc(&ffn_weight.down.data, hidden_units * inter_size);
}
//model file type用来控制loadweightfrombin的第二个模板类型参数T_IN,如果T_IN和第一个模板参数不一致，需要将T_IN的weight使用ptx cast转换为T
template<typename T>
void LlamaLayerWeight<T>::loadWeights(std::string weight_path, WeightType weight_type)
{
    loadWeightFromBin<T, T>(attn_norm_weight.gamma, {hidden_units}, weight_path + ".attention_norm.weight");
    loadWeightFromBin<T, T>(ffn_norm_weight.gamma, {hidden_units}, weight_path + ".ffn_norm.weight");

    loadWeightFromBin<T, T>(self_attn_weight.qkv.data, {hidden_units, (head_num + 2 * kv_head_num) * head_size}, weight_path + ".attention.w_qkv.weight");
    loadWeightFromBin<T, T>(self_attn_weight.output.data, {hidden_units, hidden_units}, weight_path + ".attention.wo.weight");
    loadWeightFromBin<T, T>(ffn_weight.gate.data, {hidden_units, inter_size}, weight_path + ".feed_forward.w1.weight");
    loadWeightFromBin<T, T>(ffn_weight.up.data, {hidden_units, inter_size}, weight_path + ".feed_forward.w3.weight");
    loadWeightFromBin<T, T>(ffn_weight.down.data, {inter_size, hidden_units}, weight_path + ".feed_forward.w2.weight");
    if (attn_bias) {//TODO
        loadWeightFromBin<T, T>(self_attn_weight.qkv.bias, {head_num *  head_size}, weight_path + ".attention.w_qkv.bias");
        loadWeightFromBin<T, T>(self_attn_weight.output.bias, {head_num *  head_size}, weight_path + ".attention.wo.bias");
    }   
}
template<typename T>
void LlamaLayerWeight<T>::loadWeights(T* d_attn_norm_weight,
                                T* d_ffn_norm_weight,
                                T* d_qkv_weights,
                                T* d_qkv_bias,
                                T* d_output_weights,
                                T* d_output_bias,
                                T* d_ffn_down,
                                T* d_ffn_down_bias,
                                T* d_ffn_gate,
                                T* d_ffn_up)
{
    // before kernel launch, the ptr is always void*, when luanching kernel, ptr type will be cast to float* or T*
    attn_norm_weight.gamma = d_attn_norm_weight;
    ffn_norm_weight.gamma = d_ffn_norm_weight;
    self_attn_weight.qkv.data = d_qkv_weights;
    self_attn_weight.qkv.bias = d_qkv_bias;
    self_attn_weight.output.data = d_output_weights;
    self_attn_weight.output.bias = d_output_bias;
    ffn_weight.gate.data = d_ffn_gate;
    ffn_weight.up.data = d_ffn_up;
    ffn_weight.down.data = d_ffn_down;
    ffn_weight.down.bias = d_ffn_down_bias;
}
//required in linking time
// template void LlamaLayerWeight<float>::loadWeights(float*, float*, float*, float*, float*, float*, float*, float*, float*, float*);
// template void LlamaLayerWeight<half>::loadWeights(half*, half*, half*, half*, half*, half*, half*, half*, half*, half*);

template<typename T>
void freeWeights(BaseWeight<T>& weights)
{
    cudaFree(weights.data);
    if(weights.bias != nullptr) {
        cudaFree(weights.bias);
    }

    weights.data = nullptr;
    weights.bias = nullptr;
}
template<typename T>
LlamaLayerWeight<T>::~LlamaLayerWeight()
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

