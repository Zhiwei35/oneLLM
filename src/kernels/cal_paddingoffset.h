#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_fp16.h>


void launchCalPaddingoffset(//int*      h_pinned_token_num,
                            //int*      h_token_num, // calc to validate its equal to decoder_input.shape(0) which from embedding
                            int*         padding_offset, 
                            int*         cum_seqlens,
                            const int*   seq_lens, //actual input lens
                            const int    batch_size,
                            const int    max_q_len);