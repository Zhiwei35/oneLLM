// handle prompt phase attention
// opt1: flash attention
// opt2: standard attention
// opt3: memory effcient attention

// context attention layer buffer size allocation:
// qkv_buf : qkv continouns buf when no padding
        // shape = [num_tokens, qkv_head_num, head_size], 因为各句子长度不一，所以不用bs * seqlen表示
// q_buf_2 : q buffer after rebuild padding
        // shape = [bs, max_q_len, q_head_num, head_size], 因为可能涉及多轮对话，所以q和kv使用不同的变量表示seqlen
// in mha:
// qk_buf : shape = [bs, q_head_num, max_q_len, max_k_len]
// qkv_buf: shape = [bs, max_q_len, q_head_num, head size]
// kv cache: shape = [bs, kv_head_num, max_k_len, head_size]