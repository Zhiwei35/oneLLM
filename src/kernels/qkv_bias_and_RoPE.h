#include <cuda_runtime.h>
#include <cuda.h>

template<typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(T*           q_buf,
                                            T*           k_buf,
                                            T*           v_buf,
                                            T*           QKV,
                                            const T*     qkv_bias,
                                            const int*   padding_offset,
                                            const int*   history_length,
                                            const int*   input_length,
                                            const int    batch_size,
                                            const int    seq_len,
                                            const int    token_num,
                                            const int    head_num,
                                            const int    kv_head_num,
                                            const int    head_size,
                                            const int    rotary_embedding_dim,
                                            float        rotary_embedding_base,
                                            int          max_position_embeddings,
                                            bool         use_dynamic_ntk);