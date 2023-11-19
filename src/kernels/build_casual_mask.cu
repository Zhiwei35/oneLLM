#include "src/kernels/build_casual_mask.h"
// mask shape =  [bs, max_q_len, max_k_len]

// def _make_causal_mask(
//     input_ids_shape: torch.Size, dtype: torch.dtype, device: torch.device, past_key_values_length: int = 0
// ):
//     """
//     Make causal mask used for bi-directional self-attention.
//     """
//     bsz, tgt_len = input_ids_shape
//     mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min, device=device), device=device)
//     mask_cond = torch.arange(mask.size(-1), device=device)
//     mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
//     1 0 0
//     1 1 0
//     1 1 1
//     mask = mask.to(dtype)

//     if past_key_values_length > 0:
//         mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype, device=device), mask], dim=-1)
//     return mask[None, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)
//     when past_key_values_length = 2: mask.shape = [3, 3+2], and then pad 0 to the max q len and max k len
//     0 0 1 0 0
//     0 0 1 1 0
//     0 0 1 1 1  

template<typename T>
__global__ void BuildCausalMasksConsideringContextPastKV(T* mask,
                                                const int* q_lens,  //input lens, shape=[batch size]
                                                const int* k_lens,  //context lens, shape=[batch size]
                                                int max_q_len,
                                                int max_k_len){
    int tid = threadIdx.x;
    int qlen = q_lens[blockIdx.x];
    int klen = k_lens[blockIdx.x];
    mask += blockIdx.x * max_q_len * max_k_len;
    int offset = threadIdx.x;
    // note: this judgement confirms we dont exceed data boundry
    while (offset < max_q_len * max_k_len){
        int q = offset / max_k_len;
        int k = offset % max_k_len;
        bool is_one = q < qlen && k < klen && k <= q + (klen - qlen) && k >= klen - qlen;
        mask[offset] = static_cast<T>(is_one);

        offset += blockDim.x;
    }
}

template<typename T>
void launchBuildCausalMasks(T* mask, 
                            const int* q_lens, 
                            const int* k_lens, 
                            int max_q_len, 
                            int max_k_len, 
                            int batch_size)
{
    BuildCausalMasksConsideringContextPastKV<<<batch_size, 256>>>(mask, q_lens, k_lens, max_q_len, max_k_len);
}

template void launchBuildCausalMasks(float* mask, const int*, const int*, int, int, int);
