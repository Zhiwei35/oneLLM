#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "src/layers/decoder/context_decoder.h"

int main(int argc, char** argv)
{
    int head_num = 4;
    int kv_head_num = 2;
    int head_size = 8;
    int inter_size = 12;
    int num_layers = 2;
    int max_seq_len = 12;
    int hidden_units = (head_num + 2 * kv_head_num) * head_size;
    int q_hidden_units = head_num * head_size;
    float rmsnorm_eps = 1e-16;
    LLaMAAttentionStaticParams attn_static_params;
    attn_static_params.rotary_embedding_dim = 128;
    attn_static_params.rotary_embedding_base = 10000;
    attn_static_params.max_position_embeddings = 2048;
    attn_static_params.use_dynamic_ntk = false; // for dyn scaling rope
    LLaMAAttentionDynParams attn_dyn_params;
    attn_dyn_params.batch_size = 2;
    attn_dyn_params.num_tokens = 14;
    attn_dyn_params.max_q_len = 8;
    attn_dyn_params.max_k_len = max_seq_len;
    bool is_free_buffer_after_fwd = true;

    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    cudaStream_t stream;
    cublasCreate(&cublas_handle);
    cublasSetMathMode(cublas_handle, CUBLAS_DEFAULT_MATH);
    cublasWrapper* cublas_wrapper = new cublasWrapper(cublas_handle, cublaslt_handle);
    BaseAllocator* allocator = new CudaAllocator;

    float* h_decoder_input = (float*) malloc(sizeof(float) * hidden_units * attn_dyn_params.num_tokens);
    float* d_decoder_input;
    cudaMalloc((void**)&d_decoder_input, sizeof(float) * hidden_units * attn_dyn_params.num_tokens);
    for(int i = 0; i < hidden_units * attn_dyn_params.num_tokens; i++) { 
       h_decoder_input[i] = 1.0f;
    }

    float* h_mask = (float*) malloc(sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len);
    float* d_mask;
    cudaMalloc((void**)&d_mask, sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len);
    for(int i = 0; i < attn_dyn_params.max_q_len * attn_dyn_params.max_k_len * attn_dyn_params.batch_size; i++){
        h_mask[i] = 1.0f;
    }

    //max_seq_len is the max kv cache len
    float* h_all_k_cache = (float*) malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);
    float* d_all_k_cache;
    cudaMalloc((void**)&d_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    float* h_all_v_cache = (float*) malloc(sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);
    float* d_all_v_cache;
    cudaMalloc((void**)&d_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size; i++) {
       h_all_k_cache[i] = 1.0f;
       h_all_v_cache[i] = 1.0f;
    }
    // padding to max_q_len
    int* h_padding_offset = (int*) malloc(sizeof(int) * attn_dyn_params.num_tokens);
    int* d_padding_offset;
    cudaMalloc((void**)&d_padding_offset, sizeof(int) * attn_dyn_params.num_tokens);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < attn_dyn_params.num_tokens; i++) { // 3
       h_padding_offset[i] = i < 7 ? 0 : 1;// two seqlens are both 7, tokens num=14
    }
    int* h_history_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_history_len;
    cudaMalloc((void**)&d_history_len, sizeof(int) * attn_dyn_params.batch_size);
    int* h_input_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_input_len;
    cudaMalloc((void**)&d_input_len, sizeof(int) * attn_dyn_params.batch_size);
    int* h_ctx_len = (int*) malloc(sizeof(int) * attn_dyn_params.batch_size);
    int* d_ctx_len;
    cudaMalloc((void**)&d_ctx_len, sizeof(int) * attn_dyn_params.batch_size);
    for(int i = 0; i < attn_dyn_params.batch_size; i++){
        h_history_len[i] = 0; // for kv cache cumsum seqlen and rope's timestep compute
        h_input_len[i] = 7; // corresponding to padding offset
        h_ctx_len[i] = h_history_len[i] + h_input_len[i];
    }
    // weight
    // this weight is belong to llamaweight
    float* h_output_norm_weight = (float*)malloc(sizeof(float) * hidden_units);
    float* d_output_norm_weight;
    cudaMalloc((void**)&d_output_norm_weight, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units; i++){
        h_output_norm_weight[i] = 2.0f;
    }

    float* h_attn_norm_weight = (float*)malloc(sizeof(float) * hidden_units);
    float* d_attn_norm_weight;
    cudaMalloc((void**)&d_attn_norm_weight, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units; i++){
        h_attn_norm_weight[i] = 1.0f;
    }

    float* h_ffn_norm_weight = (float*)malloc(sizeof(float) * hidden_units);
    float* d_ffn_norm_weight;
    cudaMalloc((void**)&d_ffn_norm_weight, sizeof(float) * hidden_units);
    for(int i = 0; i < hidden_units; i++){
        h_ffn_norm_weight[i] = 1.0f;
    }

    float* h_qkv_weights = (float*) malloc(sizeof(float) * hidden_units * hidden_units);
    float* d_qkv_weights;
    cudaMalloc((void**)&d_qkv_weights, sizeof(float) * hidden_units * hidden_units);
    for(int i = 0; i < hidden_units * hidden_units; i++) { 
       h_qkv_weights[i] = 1.0f;
    }

    float* h_qkv_bias = (float*) malloc(sizeof(float) * (2 * kv_head_num + head_num)   * head_size);
    float* d_qkv_bias;
    cudaMalloc((void**)&d_qkv_bias, sizeof(float) * (2 * kv_head_num + head_num)  * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < (2 * kv_head_num + head_num)  * head_size; i++){
        h_qkv_bias[i] = 2.0f;
    }

    float* h_output_weights = (float*) malloc(sizeof(float) * q_hidden_units * q_hidden_units);
    float* d_output_weights;
    cudaMalloc((void**)&d_output_weights, sizeof(float) * q_hidden_units * q_hidden_units);
    for(int i = 0; i < q_hidden_units * q_hidden_units; i++) { 
       h_output_weights[i] = 1.0f;
    }

    float* h_out_bias = (float*) malloc(sizeof(float) * head_num* head_size);
    float* d_out_bias;
    cudaMalloc((void**)&d_out_bias, sizeof(float) * head_num * head_size);// wehn add bias to k, we ensure head_id < kv_head_num
    for(int i = 0; i < head_num * head_size; i++){
        h_out_bias[i] = 2.0f;
    }
    float* d_ffn_gate, *d_ffn_up, *d_ffn_down;
    float* h_ffn_gate = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    float* h_ffn_up = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    float* h_ffn_down = (float*) malloc(sizeof(float) * hidden_units * inter_size);
    cudaMalloc((void**)&d_ffn_gate, sizeof(float) * hidden_units * inter_size);
    cudaMalloc((void**)&d_ffn_up, sizeof(float) * hidden_units * inter_size);
    cudaMalloc((void**)&d_ffn_down, sizeof(float) * hidden_units * inter_size);
    for(int i = 0; i < hidden_units * inter_size; i++){
        h_ffn_gate[i] = 2.0f;
        h_ffn_up[i] = 1.0f;
        h_ffn_down[i] = 2.0f;
    }    
    // h2d
    cudaMemcpy(d_decoder_input, h_decoder_input, sizeof(float) * hidden_units * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_k_cache, h_all_k_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_all_v_cache, h_all_v_cache, sizeof(float) * num_layers * attn_dyn_params.batch_size * kv_head_num * max_seq_len * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_padding_offset, h_padding_offset, sizeof(int) * attn_dyn_params.num_tokens, cudaMemcpyHostToDevice);
    cudaMemcpy(d_history_len, h_history_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ctx_len, h_ctx_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_len, h_input_len, sizeof(int) * attn_dyn_params.batch_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, sizeof(float) * attn_dyn_params.batch_size * attn_dyn_params.max_q_len * attn_dyn_params.max_k_len, cudaMemcpyHostToDevice);

    cudaMemcpy(d_output_norm_weight, h_output_norm_weight, sizeof(float) * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_attn_norm_weight, h_attn_norm_weight, sizeof(float) * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_norm_weight, h_ffn_norm_weight, sizeof(float) * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_weights, h_qkv_weights, sizeof(float) * hidden_units * hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_qkv_bias, h_qkv_bias, sizeof(float) * (2 * kv_head_num + head_num) * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_output_weights, h_output_weights, sizeof(float) * q_hidden_units * q_hidden_units, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out_bias, h_out_bias, sizeof(float) * head_num * head_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_down, h_ffn_down, sizeof(float) * hidden_units * inter_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_gate, h_ffn_gate, sizeof(float) * hidden_units * inter_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ffn_up, h_ffn_up, sizeof(float) * hidden_units * inter_size, cudaMemcpyHostToDevice);

    DataType type = getTensorType<float>(); // note: the type should be as a class data member!
    DataType type_int = getTensorType<int>();
    std::vector<LlamaLayerWeight*> layerWeights;
    WeightType wtype = getWeightType<float>();
    layerWeights.reserve(num_layers);
    for(int i = 0; i < num_layers; i++) {
        layerWeights[i] = new LlamaLayerWeight(head_num, kv_head_num,
                                               head_size, hidden_units,
                                               inter_size, wtype,
                                               /*attn_bias*/true);
        layerWeights[i]->loadWeights<float>(d_attn_norm_weight,
                                            d_ffn_norm_weight,
                                            d_qkv_weights,
                                            d_qkv_bias,
                                            d_output_weights,
                                            d_out_bias,
                                            d_ffn_down,
                                            d_ffn_gate,
                                            d_ffn_up);
    }

    TensorMap decoder_inputs{
        {"decoder_input", Tensor(GPU, type, {attn_dyn_params.num_tokens, hidden_units}, d_decoder_input)},
        {"qkv_bias", Tensor(GPU, type, {head_num * head_size}, d_qkv_bias)},
        {"padding_offset", Tensor(GPU, type_int, {attn_dyn_params.num_tokens}, d_padding_offset)},
        {"history_length", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, d_history_len)},
        {"input_length", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, d_input_len)},
        {"context_length", Tensor(GPU, type_int, {attn_dyn_params.batch_size}, d_ctx_len)},
        {"attention_mask", Tensor(GPU, type, {attn_dyn_params.batch_size, attn_dyn_params.max_q_len, attn_dyn_params.max_k_len}, d_mask)},
        {"output_norm_weight", Tensor(GPU, type, {hidden_units}, d_output_norm_weight)};//located at llamaweights class, rather not llamalayerweigths
    };
    TensorMap decoder_outputs{
        {"decoder_output", Tensor(GPU, type, {attn_dyn_params.num_tokens, q_hidden_units}, d_decoder_output)},
        {"all_k_cache", Tensor(GPU, type,{num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_k_cache)},
        {"all_v_cache", Tensor(GPU, type, {num_layers, attn_dyn_params.batch_size, kv_head_num, max_seq_len, head_size}, d_all_v_cache)}
    };

    LlamaContextDecoder* ctxDecoder = new LlamaContextDecoder(head_num,
                                                            kv_head_num,
                                                            head_size,
                                                            inter_size,
                                                            num_layers,
                                                            attn_static_params,
                                                            rmsnorm_eps,
                                                            stream,
                                                            cublas_wrapper,
                                                            allocator,
                                                            is_free_buffer_after_fwd)
    ctxDecoder->forward(decoder_inputs, layerWeights, decoder_outputs);
    cudaDeviceSynchronize();
    // gpu buffer can be released in corresponding class
    free(h_decoder_input);
    cudaFree(d_decoder_input);
    free(h_all_k_cache);
    cudaFree(d_all_k_cache);
    free(h_all_v_cache);
    cudaFree(d_all_v_cache);
    free(h_padding_offset);
    cudaFree(d_padding_offset);
    free(h_history_len);
    cudaFree(d_history_len);
    free(h_ctx_len);
    cudaFree(d_ctx_len);
    free(h_input_len);
    cudaFree(d_input_len);
    free(h_decoder_input);
    cudaFree(d_decoder_input);
    free(h_mask);
    cudaFree(d_mask); 
    free(h_output_norm_weight);
    cudaFree(d_output_norm_weight);
    free(h_attn_norm_weight);
    free(h_ffn_norm_weight);
    free(h_qkv_weights);
    free(h_qkv_bias);
    free(h_output_weights);
    free(h_out_bias);
    free(h_ffn_down);
    free(h_ffn_up);    
    free(h_ffn_gate);
}