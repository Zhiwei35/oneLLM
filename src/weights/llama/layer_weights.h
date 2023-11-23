#pragma once
#include "src/weights/llama/norm_weights.h"
#include "src/weights/llama/attention_weights.h"
#include "src/weights/llama/ffn_weights.h"
#include "src/utils/cuda_utils.h"

struct LlamaLayerWeight {
private:
    int     head_num;
    int     kv_head_num;
    int     head_size;
    int     hidden_units;
    int     inter_size;
    WeightType weight_type;
    int     bit_size;
    bool    attn_bias;

public:
    LlamaLayerWeight() = delete;
    LlamaLayerWeight(int head_num,
                    int  kv_head_num,
                    int  head_size,
                    int  inter_size,
                    WeightType weight_type,
                    //int  group_size, //used for per_group quantization
                    bool attn_bias);// cudamalloc to weights
    ~LlamaLayerWeight();

    void loadWeights(std::string weight_path, WeightType weight_type);

    LayerNormWeight attn_norm_weight;
    LayerNormWeight ffn_norm_weight;
    LLaMAattentionWeights self_attn_weight;
    LLaMAFFNWeights ffn_weight;
};