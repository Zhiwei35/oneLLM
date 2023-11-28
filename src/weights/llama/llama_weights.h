#pragma once
#include <string>
#include "src/weights/weight.h"
#include "src/weights/base_weights.h"
#include "src/weights/llama/embedding_weights.h"
#include "src/weights/llama/layer_weights.h"

struct LlamaWeight : public Weight {
private: 
    int     hidden_units;
    int     inter_size;
    int     vocab_size;
    int     vocab_size_padded;
    int     num_layer;
    WeightType weight_type;
    std::vector<LlamaLayerWeight*> llama_layer_weight;
    LayerNormWeight out_rmsnorm_weight;
    EmbeddingWeight post_decoder_embedding_weight;
    EmbeddingWeight pre_decoder_embedding_weight;

public:    
    LlamaWeight() = default;
    LlamaWeight(
        int     head_num,
        int     kv_head_num,
        int     head_size,
        int     inter_size,
        int     vocab_size,
        int     num_layer,
        bool    attn_bias,
        WeightType weight_type       
    );
    ~LlamaWeight();
    void loadWeights(std::string weight_path);
};