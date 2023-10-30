#include <cub.h>
#include <float.h> //FLT_MIN
#include <cuda.h>
#include <iostream>
//#include <utils/gpu_config.h>
#include "src/kernels/beamsearch_topK.h"

// a b两个topK reduce输出一个topK
__device__ topK<K> reduce_functor(const topK<K>& a, const topK<K>& b) {
    topK<K> res = a;
    for(int i = 0; i < K; i++){
        res.insert(b.val[i], b.id[i]);
    }
    return res;
}
// gridsize:bs * beamwidth * BlockPerBeam 
// blocksize:256
// shape infer: [bs, beamwidth, vocab size]=>[bs, beamwidth, BlockPerBeam, K]
template<int K, int blockSize, int BlockPerBeam>
__global__ void topK_kernel_round1(const float* probs, const int vocab_size,
                                         int* topK_ids, float* topK_vals)
{
    typedef cub::BlockReduce<topK<K>, blockSize> blockreduce;
    __shared__ typename blockreduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row_id = bid / BlockPerBeam;
    int block_lane = bid % BlockPerBeam;
    topK<K> thread_topK;
    thread_topK.init();
    // thread local reduce
    for(int data_id = tid + blocklane * blockSize; data_id < vocab_size; data_id += BlockPerBeam * blockSize){
        int data_offset = data_id + row_id * vocab_size;
        float data = probs[data_offset];
        thread_topK.insert(data, data_id);
    }
    //block local reduce
    topK<K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor);

    if(tid == 0){
        for(int k_offset = 0; k_offset < K; k_offset++) {
            // topK_vals[row_id * vocab_size + block_lane * blockSize + k_offset] = block_topK.val[k_offset];
            topK_vals[row_id * BlockPerBeam + block_lane * K + k_offset] = block_topK.val[k_offset];
            topK_ids[row_id * BlockPerBeam + block_lane * K + k_offset] = block_topK.id[k_offset];//索引offset要根据output buffer的shape来计算

        }
    }
    //__syncthreads();
}
//[bs, beamwidth, BlockPerBeam, K] => [bs, beamwidth, K] ?感觉这里应该是[bs, K]?如果是，明天来修改代码，ids也应该是beamwidth*vocalsize中的全局word id
//gridSize = bs
//blockSize = 256
template<int K, int blockSize, int BlockPerBeam>
__global__ void topK_kernel_round2(const int* topK_ids, const float* topK_vals, 
                                    int* final_topK_ids, float* final_topK_vals)
{
    typedef cub::BlockReduce<topK<K>, blockSize> blockreduce;
    __shared__ typename blockreduce::TempStorage temp_storage;

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int row_id = bid;
    topK<K> thread_topK;
    for(int i = tid; i < BlockPerBeam * K; i += blockDim.x) {
        int data_offset = bid * K * BlockPerBeam * K + tid;
        thread_topK.insert(topK_vals[data_offset], topK_ids[data_offset]);
    }
    //下一行应该求出来的block_topK就是K个值了，不需要在这外面走K次for一个个求吧
    topK<K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor);
    if(tid == 0){
        for(int k_offset = 0; k_offset < K; k_offset++) {
            // topK_vals[row_id * vocab_size + block_lane * blockSize + k_offset] = block_topK.val[k_offset];
            final_topK_vals[bid * K + k_offset] = block_topK.val[k_offset];
            final_topK_ids[bid * K + k_offset] = block_topK.id[k_offset];//索引offset要根据output buffer的shape来计算
        }
    }    
}

void launchTopKforBeamSearch(const float* probs, 
                            const int batch_size,
                            const int vocab_size, 
                            const int beamwidth,
                            float* topk_workspace) {//GPU workspace is for temp buffer and output buffer
    constexpr int maxBlockPerBeam = 8;
    //buffer size
    int topK_val_buf_size = batch_size * beamwidth * maxBlockPerBeam * beamwidth;
    int topK_ids_buf_size = batch_size * beamwidth * maxBlockPerBeam * beamwidth;
    // memory plan
    float* topK_vals = topk_workspace;
    int* topK_ids = (int*)(topK_vals + topK_val_buf_size);
    float* final_topK_vals = (float*) (topK_ids + topK_ids_buf_size);
    int* final_topK_ids = (int*)(final_topK_vals + topK_val_buf_size);    
    // prepare launch
    // GPUConfig config;
    // TODO: how to alloc block nums more flexable according to shape
    constexpr int BlockPerBeam = std::min(vocab_size / 256 / 2, maxBlockPerBeam);
    //int maxBlockNums = config.getMaxBlockNums();
    int maxBlockNums = 1024;
    int BlockNums1 = std::min(batchsize * beamwidth * BlockPerBeam, maxBlockNums);
    int BlockNums2 = std::min(batchsize * beamwidth, maxBlockNums);
    dim3 grid_round1(BlockNums1);
    dim3 block_round1(256);
    dim3 grid_round2(BlockNums2);
    dim3 block_round2(256);
    topK_kernel_round1<beamwidth, 256, BlockPerBeam>
                        <<<grid_round1, block_round1>>>(probs, vocab_size, topK_ids, topK_vals);
    topK_kernel_round2<beamwidth, 256, BlockPerBeam>
                        <<<grid_round2, block_round2>>>(topK_ids, topK_vals, final_topK_ids, final_topK_vals);
}