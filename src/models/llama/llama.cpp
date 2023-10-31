prepare weights

Embedding(inputIds, this->weight["model.embed_tokens.weight"], hiddenStates);
RMSNorm(hiddenStates, this->weight["model.layers." + std::to_string(i) + ".input_layernorm.weight"],1e-6, attenInput);
fusedQKVLinear or 3 linear
fetch past kv cache and concat kv cache to pastkv(pastkv原本是空的Data，在prompt结束阶段才和kv concat成有数据的，随后generation再一直和当前step kv concat，和FT如出一辙)
//masked attn
RoPE
gemv(q*k^T=S)
attention_mask
softmax(get P)
gemm(P*V = O)
attn_out_linear
addbiasresidual(add embedding output)
PostRMSNorm
//MLP
linearBias ([bs, num heads, 6 * head dim])
SwiGlu
LinearBias([bs, num heads, 3 * head dim]=>[bs, num heads, head dim])
addbiasresidual(add MLP input)
RMSNorm
//search or sampling
outputembeddinglinearbias/LM head ([bs, hiddensize]=>[bs, vocabsize] or [bs, beamwidth, hiddensize]=>[bs, beamwidth, vocabsize])

///1.topK sample
topK [bs, k]
softmax [bs, k]
curand sample [bs, 1]
///2.beam search
topK ([bs, beamwidth, vocabsize]==(每行vocabsize由blocks_per_beam个block计算，得到这么多个topK)>>[bs, beamwidth, blocks_per_beam, k]==reduce得到最后的topK>>[bs, beamwidth, k])//k应该是等于beam width的
update 
update kv cache

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