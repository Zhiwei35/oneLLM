#include "src/kernels/cal_paddingoffset.h"
// shape:
    //seq_lengths:[batch size]
    //cum_seqlens:[batch size + 1],first ele is 0
    //padding_offset:[batch size * max q len]
// note: the point is to calc padding offset and cum offset
// we first use serial algo, then enhance to scan algo

__global__ void CalPaddingoffset(int*      h_token_num, // calc to validate its equal to decoder_input.shape(0) which from embedding
                                int*         padding_offset, 
                                int*         cum_seqlens,
                                const int*   seq_lens, //actual input lens
                                const int    batch_size,
                                const int    max_q_len) {
    int ind = 0;
    int cum_offset = 0;
    int total_seqlen = 0;
    for(int b = 0; b < batch_size; b++) {
        int seqlen = seq_lens[i];
        cum_seqlens[b] = total_seqlen;
        // each token in one seq has same cum offset
        for (int i = 0; i < seqlen; i++) {
            padding_offset[ind] = cum_offset;
            ind++;
        }
        cum_offset += max_q_len - seqlen;
        total_seqlen += seqlen;
    }
    cum_seqlens[batch_size] = total_seqlen;
    h_token_num[0] = total_seqlen;
}

void launchCalPaddingoffset(int*      h_pinned_token_num,
                            int*      h_token_num, // calc to validate its equal to decoder_input.shape(0) which from embedding
                            int*         padding_offset, 
                            int*         cum_seqlens,
                            const int*   seq_lens, //actual input lens
                            const int    batch_size,
                            const int    max_q_len){
    h_pinned_token_num[0] = 0;
    CalPaddingoffset<<<1, 1>>>( // question: pinned memory can be accessed by GPU directly?
        h_pinned_token_num, padding_offset, cum_seqlens, seq_lens, batch_size, max_q_len
    )
    h_token_num[0] = h_pinned_token_num[0]
}

