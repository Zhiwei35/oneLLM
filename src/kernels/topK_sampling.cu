#include <stdio.h>
#include "src/kernels/topK_sampling.h"
// mini-softmax + curand_sample
// input: [bs, K] from topK output
// output: [bs]
template<typename T>
__global__ void SamplingKernel(int* topk_id,
                               T* topk_val, //[bs,K] from topK
                               int* output_id, //[bs]
                               int* seqlen, //cu seqlen,[bs]
                               bool* is_finished, //[bs]
                               int K,
                               int rand_num, // that is step
                               int end_id, // when initialize llama model, we will init it, and this is a fixed val
                               int vocab_size)
{
    int batch_id = blockIdx.x;
    int bid = batch_id;
    int tid = threadIdx.x;
    int offset = batch_id * K + tid;
    if (tid == 0) {
        printf("topk_val max = %f\n", (float)topk_val[batch_id * K]);
        printf("topk_id[0] = %d\n", topk_id[0]);
    }

    float max_val = (float)(topk_val[batch_id * K]) ; // max val is the top of the buffer, because topK
    topk_val[offset] = expf((float)topk_val[offset] - max_val);

    __shared__ float thredhold, sum;
    if(tid == 0) {
        sum = 0.0f;
        for(int i = 0; i < K; i++) {
            sum += (float)topk_val[batch_id * K + i];
        }
        curandState_t state;
        curand_init((unsigned long long)rand_num,(unsigned long long)bid, (unsigned long long)0, &state);// not sure rand_num's type is suitable here or not
        thredhold = (float)curand_uniform(&state) * sum; // for a block
        printf("sampling info: \n");
        printf("sum = %f\n", sum);
        printf("thredhold = %f\n", thredhold);
        printf("top1 val = %f", (float)topk_val[0]);

        for(int i = 0; i < K; i++) {
            thredhold = thredhold - (float)topk_val[batch_id * K + i];
            if(thredhold < 0) {
                output_id[bid] = topk_id[batch_id * K + i] % vocab_size;
                break;
            }
        }
        seqlen[bid] = is_finished[bid] ? seqlen[bid] : seqlen[bid] + 1;
        is_finished[bid] = output_id[bid] == end_id ? 1 : 0;
    }
}
template<typename T>
void launchSampling(TensorWrapper<int>* topk_id,
                    TensorWrapper<T>* topk_val,
                    TensorWrapper<int>* seqlen,
                    TensorWrapper<bool>* is_finished,
                    TensorWrapper<int>* output_id,
                    TensorWrapper<int>* step,
                    IntDict& params) {
    int batch_size = topk_id->shape[0];
    int K = topk_id->shape[1];
    int vocab_size = params["vocab_size"];
    int step_ = step->getVal();
    int end_id = params["end_id"];

    dim3 grid(batch_size);
    dim3 block(K); // K is small, so directly allocate K threads is enough
    // std::cout << "calling sampling kernel" << "\n";
    SamplingKernel<T><<<grid, block>>>(
        topk_id->data,
        topk_val->data,
        output_id->data,
        seqlen->data,
        is_finished->data,
        K,
        step_,
        end_id,
        vocab_size
    );
    // std::cout << "called sampling kernel" << "\n";
                    }

template void launchSampling(TensorWrapper<int>* topk_id,
                    TensorWrapper<float>* topk_val,
                    TensorWrapper<int>* seqlen,
                    TensorWrapper<bool>* is_finished,
                    TensorWrapper<int>* output_id,
                    TensorWrapper<int>* step,
                    IntDict& params);
template void launchSampling(TensorWrapper<int>* topk_id,
                    TensorWrapper<half>* topk_val,
                    TensorWrapper<int>* seqlen,
                    TensorWrapper<bool>* is_finished,
                    TensorWrapper<int>* output_id,
                    TensorWrapper<int>* step,
                    IntDict& params);