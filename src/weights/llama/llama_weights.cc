#include <iostream>
#include "src/weights/llama/llama_weights.h"

LlamaWeight::LlamaWeight(
    int     head_num,
    int     kv_head_num,
    int     head_size,
    int     inter_size,
    int     vocab_size,
    int     num_layer,
    bool    attn_bias,
    WeightType weight_type,
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
    GPUMalloc(&post_decoder_embedding_weight.emb_table, vocab_size * hidden_units);
    GPUMalloc(&pre_decoder_embedding_weight.emb_table, vocab_size * hidden_units);
}

LlamaWeight::loadWeights(std::string weight_path) {
    //weight_path += '/';
    std::cout << "the weight path is " << weight_path <<"\n";
    loadWeightFromBin<float, float>(out_rmsnorm_weight.gamma, {hidden_units}, weight_path + ".norm.weight");
    loadWeightFromBin<float, float>(post_decoder_embedding_weight.emb_table, {vocab_size, hidden_units}, weight_path + ".tok_embeddings.weight");
    loadWeightFromBin<float, float>(pre_decoder_embedding_weight.emb_table, {vocab_size, hidden_units}, weight_path + ".output.weight");

    for (int layer = 0; layer < num_layer; ++layer) {
        llama_layer_weight[layer]->loadWeights(weight_path + "layers." + std::to_string(layer), weight_type);
    }
}

LlamaWeight<T>::~LlamaWeight()
{
    cudaFree(pre_decoder_embedding_weight.emb_table);
    cudaFree(out_rmsnorm_weight.gamma);
    cudaFree(post_decoder_embedding_weight.emb_table);

    for (auto& p : llama_layer_weight) {
        delete p;
    }
}