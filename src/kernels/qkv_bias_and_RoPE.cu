// This kernel only used in prompt phase
// 1.add bias to QKV, which has shape [batch_size, seq_len, 3, head_num, size_per_head], and
// QKV split to 3 split buffer q, k, v and transpose them to [batch_size, head_num, seq_len, size_per_head].

// 2.For q and k, apply RoPE, then send to attention.

// 3.rebuild padding to do mha

// input: qkv_buf : qkv continouns buf when no padding
        // shape = [num_tokens, qkv_head_num, head_size], 因为各句子长度不一，所以不用bs * seqlen表示
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
//         k/v shape = [bs, kv head num, seqlen, head size]
// seqlen=max_q_len
#include <math.h>
#include <stdio.h>

#include "src/kernels/qkv_bias_and_RoPE.h"

inline __device__ float2 GetRoPEfreq(int zid, int rot_embed_dim, float base, float t_step)
{
    const float inv_freq = t_step / powf(base, zid / (float)rot_embed_dim);
    return {cos(inv_freq), sin(inv_freq)};
}

// RoPE公式决定必须要做向量化
inline __device__ float2 GetRoPEres(const float2 v, const float2 coef)
{
    float2 rot_v;
    rot_v.x = coef.x * v.x - coef.y * v.y;
    rot_v.y = coef.x * v.y + coef.y * v.x;
    return rot_v;
}

inline __device__ half2 GetRoPEres(const half2 v, const float2 coef)
{
    float2 fv     = __half22float2(v);
    float2 rot_fv = GetRoPEres(fv, coef);
    return __float22half2_rn(rot_fv);
}

inline __device__ void apply_RoPE(half2& q, half2& k, int tid, int rot_embed_dim, float base, float t_step)
{
    if (2 * tid >= rot_embed_dim) {
        return;
    }
    const auto coef = GetRoPEfreq(2 * tid, rot_embed_dim, base, t_step);
    q               = GetRoPEres(q, coef);
    k               = GetRoPEres(k, coef);
}

inline __device__ void apply_RoPE(float4& q, float4& k, int tid, int rot_embed_dim, float base, float t_step){
    if(4 * tid >= rot_embed_dim){
        return;
    }


    TwoFloat2& q_ = *reinterpret_cast<TwoFloat2*>(&q); // q为float4 寄存器
    TwoFloat2& k_ = *reinterpret_cast<TwoFloat2*>(&k);
    
    float2 coef0 = GetRoPEfreq(4 * tid, rot_embed_dim, base, t_step);
    // float freq0 = timestep / powf(rotary_embedding_base, 4 * tid / (float) rotary_embedding_dim); //分子zid = 0,2,4,, headsize/2-1,对应的theta下标为0,1,2.对应的headsize维度的索引为(0,1),(2,3)
    q_.x = GetRoPEres(q_.x ,coef0);
    // rot0.x = coef0.x * q.x -  coef0.y * q.y; //q.x为x0,q.y为x1，head size维度上两个相邻
    // rot0.y = coef0.x * q.y +  coef0.y * q.x
    float2 coef1 = GetRoPEfreq(4 * tid + 2, rot_embed_dim, base, t_step);
    q_.y = GetRoPEres(q_.y ,coef1);
    // rot1.x = coef1.x * q.x -  coef1.y * q.y; //q.x为x2,q.y为x3，head size维度上两个相邻
    // rot1.y = coef1.x * q.y +  coef1.y * q.x;
    k_.x = GetRoPEres(k_.x ,coef0);
    k_.y = GetRoPEres(k_.y ,coef1);
}
template<typename T>
__global__ void add_fusedQKV_bias_transpose_kernel(T*           q_buf,
                                                    T*           k_buf,
                                                    T*           v_buf,
                                                    T*           QKV,
                                                    const T*     qkv_bias,
                                                    const int*   padding_offset, // created before qkv linear
                                                    const int*   history_length,
                                                    const int*   input_length, //actual length of each seq
                                                    const int    batch_size,
                                                    const int    seq_len, //max_seq_len to pad to
                                                    const int    token_num,
                                                    const int    head_num,
                                                    const int    kv_head_num,
                                                    const int    head_size,
                                                    const int    rotary_embedding_dim,
                                                    float        rotary_embedding_base, // default 10000 in llama
                                                    int          max_position_embeddings,/*default 2048 in llama, placeholder for ntk RoPE*/
                                                    bool         use_dynamic_ntk/*placeholder for ntk RoPE*/){
    int vec_size = Vec<T>::size;
    using Vec_t = typename Vec<T>::Type;
    int token_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];
    // 0. filter the redundant part, we'd better to allocate more threads than data to ensure all data can be vectorized
    bool is_data = tid * vec_size < head_size;
    bool is_half = sizeof(T) == 2;
    // 1. prapare rebuilding , do rebuild padding and transpose when store
    int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding

    int batch_id = dst_token_id / seq_len; //seqlen is max_seq_len for padding used to unify all seq's length
    int local_token_id = dst_token_id % seq_len; //每个seq中的局部token id

    //2. bias add
    int qkv_head_num = head_num + 2 * kv_head_num;
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size;
    int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size;
    int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size;
    // note: scalar add can be replaced by 3 overloaded function call, which is implemented by float add, float2 add and float4 add.
    // TODO: reduce the pointer converter and fuse for loop
    Vec_t q, k, v;
    if(is_data){
        q = *reinterpret_cast<Vec_t*>(&QKV[q_id]);
        Vec_t q_bias = *reinterpret_cast<Vec_t*>(const_cast<T*>(&qkv_bias[head_id * head_size + tid * vec_size]));
        for(int i = 0; i < vec_size; i++) {
            reinterpret_cast<float*>(&q)[i] += reinterpret_cast<float*>(&q_bias)[i];
        }
    }
    // note: kv judge condition is add a item that head_id<kv_head_id in case of GQA and MQA
    if(is_data && head_id < kv_head_num){
        k = *reinterpret_cast<Vec_t*>(&QKV[k_id]);
        // note: I missed a vec_size about the bias offset causing memcpyd2h misaligned address
        Vec_t k_bias =*reinterpret_cast<Vec_t*>(const_cast<T*>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size]));
        for(int i = 0; i < vec_size; i++) {
            reinterpret_cast<float*>(&k)[i] += reinterpret_cast<float*>(&k_bias)[i];
        }

        v = *reinterpret_cast<Vec_t*>(&QKV[v_id]);
        Vec_t v_bias = *reinterpret_cast<Vec_t*>(const_cast<T*>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]));
        for(int i = 0; i < vec_size; i++) {
            reinterpret_cast<float*>(&v)[i] += reinterpret_cast<float*>(&v_bias)[i];
        }
    }

    //3. RoPE
    const int cur_seq_history_len = history_length[batch_id]; // pay attention to where the history lenght cumsum
    const int context_length = cur_seq_history_len + input_length[batch_id];
    const int timestep = cur_seq_history_len + local_token_id;//+ local_token_id得到m，即要结合history length做全局位置编码
    // timestep为cos(m*theta)中的m
    
    apply_RoPE(q, k, tid, rotary_embedding_dim, rotary_embedding_base, timestep);
    //4.write back to gmem and do transpose
    // [bs, head num, seqlen, head size]
    // pay attention to local token id and kv head num and max_seq_len(seq_len)
    int dst_q_id = batch_id * seq_len * head_num * head_size + 
                            head_id * seq_len * head_size +
                                local_token_id * head_size + tid * vec_size;

    int dst_kv_id = batch_id * seq_len * kv_head_num * head_size + 
                            head_id * seq_len * head_size +
                                local_token_id * head_size + tid * vec_size;
    if(is_data){
        *reinterpret_cast<Vec_t*>(&q_buf[dst_q_id]) = q; // remember to add & before q_buf[], cause q_buf[] is a scalar
        if (head_id < kv_head_num) {//for MQA and GQA
            *reinterpret_cast<Vec_t*>(&k_buf[dst_kv_id]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dst_kv_id]) = v;
        }
    }
                                                    }

template<>
__global__ void add_fusedQKV_bias_transpose_kernel(half*           q_buf,
                                                    half*           k_buf,
                                                    half*           v_buf,
                                                    half*           QKV,
                                                    const half*     qkv_bias,
                                                    const int*   padding_offset, // created before qkv linear
                                                    const int*   history_length,
                                                    const int*   input_length, //actual length of each seq
                                                    const int    batch_size,
                                                    const int    seq_len, //max_seq_len to pad to
                                                    const int    token_num,
                                                    const int    head_num,
                                                    const int    kv_head_num,
                                                    const int    head_size,
                                                    const int    rotary_embedding_dim,
                                                    float        rotary_embedding_base, // default 10000 in llama
                                                    int          max_position_embeddings,/*default 2048 in llama, placeholder for ntk RoPE*/
                                                    bool         use_dynamic_ntk/*placeholder for ntk RoPE*/){
    int vec_size = Vec<half>::size;
    using Vec_t = typename Vec<half>::Type;
    int token_id = blockIdx.x;
    int head_id = blockIdx.y;
    int tid = threadIdx.x;
    int token_padding_offset = padding_offset[token_id];
    // 0. filter the redundant part, we'd better to allocate more threads than data to ensure all data can be vectorized
    bool is_data = tid * vec_size < head_size;
    // 1. prapare rebuilding , do rebuild padding and transpose when store
    int dst_token_id = token_id + token_padding_offset; // token id after rebuild padding

    int batch_id = dst_token_id / seq_len; //seqlen is max_seq_len for padding used to unify all seq's length
    int local_token_id = dst_token_id % seq_len; //每个seq中的局部token id

    //2. bias add
    int qkv_head_num = head_num + 2 * kv_head_num;
    int q_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size;
    int k_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size;
    int v_id = token_id * qkv_head_num * head_size + head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size;
    // note: scalar add can be replaced by 3 overloaded function call, which is implemented by float add, float2 add and float4 add.
    // TODO: reduce the pointer converter and fuse for loop
    Vec_t q, k, v;
    if(is_data){
        q = *reinterpret_cast<Vec_t*>(&QKV[q_id]);
        Vec_t q_bias = *reinterpret_cast<Vec_t*>(const_cast<half*>(&qkv_bias[head_id * head_size + tid * vec_size]));
        q = __hadd2(q, q_bias);
    }
    // note: kv judge condition is add a item that head_id<kv_head_id in case of GQA and MQA
    if(is_data && head_id < kv_head_num){
        k = *reinterpret_cast<Vec_t*>(&QKV[k_id]);
        // note: I missed a vec_size about the bias offset causing memcpyd2h misaligned address
        Vec_t k_bias =*reinterpret_cast<Vec_t*>(const_cast<half*>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size]));
        k = __hadd2(k, k_bias);
        v = *reinterpret_cast<Vec_t*>(&QKV[v_id]);
        Vec_t v_bias = *reinterpret_cast<Vec_t*>(const_cast<half*>(&qkv_bias[head_id * head_size + tid * vec_size + head_num * head_size + kv_head_num * head_size]));
        v = __hadd2(v, v_bias);
    }

    //3. RoPE
    const int cur_seq_history_len = history_length[batch_id]; // pay attention to where the history lenght cumsum
    const int context_length = cur_seq_history_len + input_length[batch_id];
    const int timestep = cur_seq_history_len + local_token_id;//+ local_token_id得到m，即要结合history length做全局位置编码
    // timestep为cos(m*theta)中的m
    
    apply_RoPE(q, k, tid, rotary_embedding_dim, rotary_embedding_base, timestep);
    //4.write back to gmem and do transpose
    // [bs, head num, seqlen, head size]
    // pay attention to local token id and kv head num and max_seq_len(seq_len)
    int dst_q_id = batch_id * seq_len * head_num * head_size + 
                            head_id * seq_len * head_size +
                                local_token_id * head_size + tid * vec_size;

    int dst_kv_id = batch_id * seq_len * kv_head_num * head_size + 
                            head_id * seq_len * head_size +
                                local_token_id * head_size + tid * vec_size;
    if(is_data){
        *reinterpret_cast<Vec_t*>(&q_buf[dst_q_id]) = q; // remember to add & before q_buf[], cause q_buf[] is a scalar
        if (head_id < kv_head_num) {//for MQA and GQA
            *reinterpret_cast<Vec_t*>(&k_buf[dst_kv_id]) = k;
            *reinterpret_cast<Vec_t*>(&v_buf[dst_kv_id]) = v;
        }
    }
                                                    }

// input: qkv_buf : qkv continouns buf when no padding
        // shape = [num_tokens, qkv_head_num, head_size], 因为各句子长度不一，所以不用bs * seqlen表示
// output: q shape = [bs, head num, seqlen, head size], if k v is this shape, maybe need tranpose in successor steps, ep in cublas
//         k/v shape = [bs, kv head num, seqlen, head size], 这里的seqlen应该是max_q_len
template<typename T>
void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<T>* q_buf,
                                           TensorWrapper<T>* k_buf,
                                           TensorWrapper<T>* v_buf,
                                           TensorWrapper<T>* QKV,
                                           BaseWeight<T>& qkv,
                                           //Tensor* qkv_bias,
                                           TensorWrapper<int>* padding_offset,
                                           TensorWrapper<int>* history_length,
                                           TensorWrapper<int>* input_length,
                                           LLaMAAttentionStaticParams& params){
    int token_num = QKV->shape[0];
    int qkv_head_num = QKV->shape[1];
    int head_size = QKV->shape[2];
    int batch_size = q_buf->shape[0];
    int head_num = q_buf->shape[1];
    int seq_len = q_buf->shape[2];
    int kv_head_num = (qkv_head_num - head_num) / 2;

    dim3 grid(token_num ,head_num);
    dim3 block((head_size / Vec<float>::size + 32 - 1) / 32 * 32);// apply 2 eles vectorization to match RoPE
    // printf("calling qkvbias and rope\n");
    add_fusedQKV_bias_transpose_kernel<T><<<grid, block>>>( q_buf->data,
                                                            k_buf->data,
                                                            v_buf->data,
                                                            QKV->data,
                                                            qkv.bias,
                                                            padding_offset->data,
                                                            history_length->data,
                                                            input_length->data,
                                                            batch_size,
                                                            seq_len,
                                                            token_num,
                                                            head_num,
                                                            kv_head_num,
                                                            head_size,
                                                            params.rotary_embedding_dim,
                                                            params.rotary_embedding_base,
                                                            params.max_position_embeddings,
                                                            params.use_dynamic_ntk);
    // printf("called qkv bias and rope\n");
}

template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<float>* q_buf,
                                           TensorWrapper<float>* k_buf,
                                           TensorWrapper<float>* v_buf,
                                           TensorWrapper<float>* QKV,
                                           BaseWeight<float>& qkv,
                                           //Tensor* qkv_bias,
                                           TensorWrapper<int>* padding_offset,
                                           TensorWrapper<int>* history_length,
                                           TensorWrapper<int>* input_length,
                                           LLaMAAttentionStaticParams& params);
template void launchAddFusedQKVBiasTransposeAndRoPE(TensorWrapper<half>* q_buf,
                                           TensorWrapper<half>* k_buf,
                                           TensorWrapper<half>* v_buf,
                                           TensorWrapper<half>* QKV,
                                           BaseWeight<half>& qkv,
                                           //Tensor* qkv_bias,
                                           TensorWrapper<int>* padding_offset,
                                           TensorWrapper<int>* history_length,
                                           TensorWrapper<int>* input_length,
                                           LLaMAAttentionStaticParams& params);