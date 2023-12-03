// def __init__(
//     self,
//     vocab_size=32000,
//     hidden_size=4096,
//     intermediate_size=11008,
//     num_hidden_layers=32,
//     num_attention_heads=32,
//     hidden_act="silu",
//     max_position_embeddings=2048,
//     initializer_range=0.02,
//     rms_norm_eps=1e-6,
//     use_cache=True,
//     pad_token_id=0,
//     bos_token_id=1,
//     eos_token_id=2,
//     tie_word_embeddings=False,
//     **kwargs,
// ):
#include "src/models/llama/llama.h"
#include "src/models/tokenizer.h"
void Llama::allocatePersistBuffer(int max_batch_size){
    output_ids_buf_ = (int*)allocator->Malloc(output_ids_buf_, sizeof(int) * max_batch_size * output_token_limit, true); //本来这里长度是session_len=4096, 定义在lmdeploy/model.py
    h_input_ids_buf_ =
        (int*)allocator->Malloc(h_input_ids_buf_, sizeof(int) * max_batch_size * session_len_, false, true);
    h_input_length_buf_ =
        (int*)allocator->Malloc(h_input_length_buf_, sizeof(int) * max_batch_size, false, true);
    h_history_length_buf_ =
        (int*)allocator->Malloc(h_history_length_buf_, sizeof(int) * max_batch_size, false, true);
    h_context_length_buf_ =
        (int*)allocator->Malloc(h_context_length_buf_, sizeof(int) * max_batch_size, false, true);
    h_sequence_lengths_ =
        (int*)allocator->Malloc(h_sequence_lengths_, sizeof(int) * max_batch_size, false, true);
    h_k_cache_ptr_buf_ =
        (uint64_t*)allocator->Malloc(h_k_cache_ptr_buf_, sizeof(uint64_t) * max_batch_size, true, true);
    h_v_cache_ptr_buf_ =
        (uint64_t*)allocator->Malloc(h_v_cache_ptr_buf_, sizeof(uint64_t) * max_batch_size, true, true);
    h_finished_buf_ = (bool*)allocator->Malloc(h_finished_buf_, sizeof(bool) * max_batch_size, false, true);
    // h_seq_limit_len_ =
    //     (uint32_t*)allocator->Malloc(h_seq_limit_len_, sizeof(uint32_t) * max_batch_size, false, true);
}

void Llama::allocateBuffer(int batch_size, int session_len)
{
    context_decoder_input_buf_ =
        (float*)allocator->Malloc(context_decoder_input_buf_, sizeof(float) * max_context_token_num_ * hidden_units, false);
    context_decoder_output_buf_ =
        (float*)allocator->Malloc(context_decoder_output_buf_, sizeof(float) * max_context_token_num_ * hidden_units, false);
    context_decoder_ids_buf_ =
        (int*)allocator->Malloc(context_decoder_ids_buf_, sizeof(int) * max_context_token_num_, false);

    decoder_input_buf_  = (T*)allocator->Malloc(decoder_input_buf_, sizeof(T) * batch_size * hidden_units, false);
    decoder_output_buf_ = (T*)allocator->Malloc(decoder_output_buf_, sizeof(T) * batch_size * hidden_units, false);

    input_ids_buf_      = (int*)allocator->Malloc(input_ids_buf_, sizeof(int) * batch_size * session_len, true);
    input_length_buf_   = (int*)allocator->Malloc(input_length_buf_, sizeof(int) * batch_size);
    history_length_buf_ = (int*)allocator->Malloc(history_length_buf_, sizeof(int) * batch_size);
    context_length_buf_ = (int*)allocator->Malloc(context_length_buf_, sizeof(int) * batch_size);
    sequence_lengths_    = (int*)allocator->Malloc(sequence_lengths_, sizeof(int) * batch_size, false);

    k_cache_ptr_buf_ = (uint64_t*)allocator->Malloc(k_cache_ptr_buf_, sizeof(uint64_t) * batch_size);
    v_cache_ptr_buf_ = (uint64_t*)allocator->Malloc(v_cache_ptr_buf_, sizeof(uint64_t) * batch_size);

    // logits_buf_       = (float*)allocator->Malloc(logits_buf_, sizeof(float) * batch_size * vocab_size, false);
    //输出id buffer
    token_ids_buf_ = (int*)allocator->Malloc(token_ids_buf_, sizeof(int) * batch_size * session_len * 2, true);

    //end_ids_buf_   = (int*)allocator->Malloc(end_ids_buf_, sizeof(int) * batch_size, false);
    finished_buf_  = (bool*)allocator->Malloc(finished_buf_, sizeof(bool) * batch_size, false);
    // seq_limit_len_ = (uint32_t*)allocator->Malloc(seq_limit_len_, sizeof(uint32_t) * batch_size, false);
}
//seems we should self define max_context_len, since we only support bs=1 now
void Llama::InitializeForContextDecoder(){
    //only support and assumed bs = 1
    CHECK(cudaMemcpy(input_ids_buf,  //
                    input_ids.getPtr<int>(), //input_ids 是Tensor
                    sizeof(int) * h_input_length_buf_[0], 
                    cudaMemcpyHostToDevice));
    h_history_length_buf_[0] = ;
    h_context_length_buf_[0] = h_input_length_buf_[0] + h_history_length_buf_[0];
    
    h_k_cache_ptr_buf_[i] = ;
    h_v_cache_ptr_buf_[i] = ;
    step = h_context_length_buf_[0];
    // batch size = 1
    CHECK(
        cudaMemcpy(input_length_buf_, h_input_length_buf_, sizeof(int) * 1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        history_length_buf_, h_history_length_buf_, sizeof(int) * 1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        context_length_buf_, h_context_length_buf_, sizeof(int) * 1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        k_cache_ptr_buf_, h_k_cache_ptr_buf_, sizeof(uintptr_t) * 1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        v_cache_ptr_buf_, h_v_cache_ptr_buf_, sizeof(uintptr_t) * 1, cudaMemcpyHostToDevice));

}

void Llama::InitializeForSelfDecoder(){
    CHECK(cudaMemcpy(
        context_length_buf_, h_context_length_buf_, sizeof(int) * 1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        k_cache_ptr_buf_, h_k_cache_ptr_buf_, sizeof(uintptr_t) * 1, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(
        v_cache_ptr_buf_, h_v_cache_ptr_buf_, sizeof(uintptr_t) * 1, cudaMemcpyHostToDevice));
    //搜一下d2d怎么copy
    CHECK(
        cudaMemcpy(sequence_lengths_, context_length_buf_, sizeof(int) * 1, cudaMemcpyHostToDevice));
    //step_ = max_context_len_;

    // 可自定义输出长度
    h_finished_buf_[0] = sequence_lengths_[0] >= max_seq_len;
    CHECK(
        cudaMemcpy(finished_buf_, h_finished_buf_, sizeof(bool) * 1, cudaMemcpyHostToDevice));

}

std::string Llama::MakeInput(const std::string &history, int round, const std::string &input) {
    return (round == 0 ? pre_prompt : history) + user_role + input + bot_role;
}

std::string Llama::MakeHistory(const std::string &history, int round, const std::string &input, const std::string &output) {
    return (round == 0 ? pre_prompt : history) + user_role + input + bot_role + output + history_sep;
}
void Llama::inputEmbedding(){
    launchInputEmbedding(input_ids, embed_table, decoder_input);
}
void Llama::contextDecode(){
    context_decoder->forward(decoder_inputs,
                             layerWeights,
                             decoder_outputs, 
                             attn_dyn_params);
}
void Llama::selfDecoder(){
    self_decoder->forward(decoder_inputs,
                          layerWeights,
                          decoder_outputs,
                          attn_dyn_params);
}
int Llama::LMHeadAndTopKSample(){
    launchLinearGemm(/*Tensor**/ decoder_output, //[bs, hidden_units]
                    /*BaseWeight&*/ weight, //[hidden_units, vocab_size]
                     /*Tensor**/ probs);
    launchTopKforBeamSearch(probs, // [bs, vocab_size] 
                            batch_size,
                            vocab_size, 
                            topk_workspace);//output
    launchSampling(/*Tensor**/ topk_id, // in
                   /*Tensor**/ topk_val,//in
                   /*Tensor**/ seqlen,//out
                   /*Tensor**/ is_finished,//out
                   /*Tensor**/ output_id, //out
                   /*IntDict&*/params); //in, including step vocabsize endid
}
int Llama::forward(TensorMap& inputs, TensorMap& outputs, Weight& weight, DynParams& dparams, StaticParams& sparams){
    inputEmbedding();
    contextDecode();
    selfDecoder();
    int res = LMHeadAndTopKSample();
    return res;   
}
// single request response, batch size = 1
std::string Llama::Response(const std::string &input, CallBack PrintRes) {
    Tokenizer tokenizer;
    // this input already include self-defined pre prompt
    Tensor input_ids = tokenizer.Encode(input);

    // std::vector <float> ids;
    // for (int i = 0; i < input_ids.Count(0); i++) {
    //     ids.push_back(((float*)input_ids.data)[i]);
    // }
    // int seqLen = ids.size();
    // inputIds.CopyFrom(Data(DataType::FLOAT32, {1, seqLen}, ids));
    
    // ensure prepared all needed input buffer
    int index = 0;
    std::string retString = "";
    while (true) {
        // kv cache here is empty, only buffer
        // TODO move all needed data to GPU
        // no need input attnmask and positionid like fastllm, cause we build attnmask and dont support abili now
        int ret = Forward(input_ids, all_k_cache, all_v_cache);
        if (ret == eos_token_id) {
            break;
        }

        //results.push_back(ret);
        std::string genString = tokenizer.Decode(Tensor(Device::GPU, DataType::FP32, {1}, &ret)).c_str();
        retString += curString;
        PrintRes(index, genString.c_str());
        index++;
        if (index == output_token_limit) {
            break;
        }
        // deep copy
        input_ids = Tensor(Device::GPU, DataType::FP32, {1, 1}, &ret);
    }
    PrintRes(-1, retString.c_str());
    return retString;
}
