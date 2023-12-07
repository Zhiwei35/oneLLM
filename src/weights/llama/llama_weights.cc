#include <iostream>
#include "src/weights/llama/llama_weights.h"
template<typename T>
LlamaWeight<T>::LlamaWeight(
    int     head_num,
    int     kv_head_num,
    int     head_size,
    int     inter_size,
    int     vocab_size,
    int     num_layer,
    bool    attn_bias,
    WeightType weight_type
):
    hidden_units(head_num * head_size),
    inter_size(inter_size),
    vocab_size(vocab_size),
    vocab_size_padded(vocab_size),
    num_layer(num_layer),
    weight_type(weight_type) 
{
    llama_layer_weight.reserve(num_layer);
    for (int l = 0; l < num_layer; ++l) {
        llama_layer_weight.push_back(new LlamaLayerWeight<T>(head_num,
                                                            kv_head_num,
                                                            head_size,
                                                            inter_size,
                                                            weight_type,
                                                            //group_size,
                                                            attn_bias));
    }
    GPUMalloc(&out_rmsnorm_weight.gamma, hidden_units);
    GPUMalloc(&post_decoder_embedding_weight.data, vocab_size * hidden_units);
    GPUMalloc(&pre_decoder_embedding_weight.data, vocab_size * hidden_units);
    pre_decoder_embedding_weight.shape = {vocab_size, hidden_units};
    post_decoder_embedding_weight.shape = {hidden_units, vocab_size};
    pre_decoder_embedding_weight.type = weight_type;
    post_decoder_embedding_weight.type = weight_type;
}
template<typename T>
void LlamaWeight<T>::loadWeights(std::string weight_path) {
    //weight_path += '/';
    std::cout << "the weight path is " << weight_path <<"\n";
    // weight from HF is always half type
    loadWeightFromBin<T, half>(out_rmsnorm_weight.gamma, {hidden_units}, weight_path + ".norm.weight");
    loadWeightFromBin<T, half>(post_decoder_embedding_weight.data, {vocab_size, hidden_units}, weight_path + ".tok_embeddings.weight");
    loadWeightFromBin<T, half>(pre_decoder_embedding_weight.data, {vocab_size, hidden_units}, weight_path + ".output.weight");
    for (int layer = 0; layer < num_layer; ++layer) {
        llama_layer_weight[layer]->loadWeights(weight_path + "layers." + std::to_string(layer), weight_type);
    }
}

template<typename T>
void LlamaWeight<T>::loadWeightsFromDummy() {

}

template<typename T>
LlamaWeight<T>::~LlamaWeight()
{
    cudaFree(pre_decoder_embedding_weight.data);
    cudaFree(out_rmsnorm_weight.gamma);
    cudaFree(post_decoder_embedding_weight.data);

    for (auto& p : llama_layer_weight) {
        delete p;
    }
}
