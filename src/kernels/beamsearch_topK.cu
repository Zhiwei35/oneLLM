#include <float.h> //FLT_MIN
#include <cuda.h>
#include <iostream>
//#include <utils/gpu_config.h>
#include "src/kernels/beamsearch_topK.h"
#include <cub/cub.cuh>

// Note: a b两个topK reduce输出一个topK
template<int K>
__device__ topK<K> reduce_functor(const topK<K>& a, const topK<K>& b) {
    topK<K> res = a;
    for(int i = 0; i < K; i++){
        res.insertHeap(b.val[i], b.id[i]);
    }
    return res;
}
// gridsize:bs * beamwidth * BlockPerBeam 
// blocksize:256
// shape infer: [bs, beamwidth, vocab size] => [bs, beamwidth, BlockPerBeam, K]
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
    for(int data_id = tid + block_lane * blockSize; data_id < vocab_size; data_id += BlockPerBeam * blockSize){
        int data_offset = data_id + row_id * vocab_size;
        float data = probs[data_offset];
        thread_topK.insertHeap(data, data_offset);
       //thread_topK.insertHeap(data, data_id); // bug
    }
    //block local reduce
    topK<K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor<K>);

    if(tid == 0){
        for(int k_offset = 0; k_offset < K; k_offset++) {
            // topK_vals[row_id * vocab_size + block_lane * blockSize + k_offset] = block_topK.val[k_offset]; //bug
            topK_vals[row_id * BlockPerBeam * K + block_lane * K + k_offset] = block_topK.val[k_offset];
            topK_ids[row_id * BlockPerBeam * K  + block_lane * K + k_offset] = block_topK.id[k_offset];//output offset要根据output buffer的shape来计算

        }
    }
}
// shape infer: [bs, beamwidth, BlockPerBeam, K] => [bs, K]
// ids是beamwidth * vocalsize中的全局word id
// gridSize = bs
// blockSize = 256
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
    // thread local reduce    
    for(int i = tid; i < K * BlockPerBeam * K; i += blockDim.x) {
        int data_offset = bid * K * BlockPerBeam * K + i;
        thread_topK.insertHeap(topK_vals[data_offset], topK_ids[data_offset]);
    }
    // block reduce
    topK<K> block_topK = blockreduce(temp_storage).Reduce(thread_topK, reduce_functor<K>);
    if(tid == 0){
        for(int k_offset = 0; k_offset < K; k_offset++) {
            // topK_vals[row_id * vocab_size + block_lane * blockSize + k_offset] = block_topK.val[k_offset]; //bug
            final_topK_vals[bid * K + k_offset] = block_topK.val[k_offset];
            final_topK_ids[bid * K + k_offset] = block_topK.id[k_offset];
        }
    }    
}

void launchTopKforBeamSearch(const float* probs, 
                            const int batch_size,
                            const int vocab_size, 
                            float* topk_workspace) {// GPU workspace is for intermediate buffer and output buffer
    constexpr int maxBlockPerBeam = 8;
    constexpr int beamwidth = 2;
    // buffer size
    int topK_val_buf_size = batch_size * beamwidth * maxBlockPerBeam * beamwidth;
    int topK_ids_buf_size = batch_size * beamwidth * maxBlockPerBeam * beamwidth;
    int final_topK_val_buf_size = batch_size * beamwidth;
    // memory plan
    float* topK_vals = topk_workspace;
    int* topK_ids = (int*)(topK_vals + topK_val_buf_size);
    float* final_topK_vals = (float*) (topK_ids + topK_ids_buf_size);
    int* final_topK_ids = (int*)(final_topK_vals + final_topK_val_buf_size);    
    // prepare launch
    // TODO: add GPUconfig API to easily get GPU config, ep: maxblocknums
    // GPUConfig config;
    // int maxBlockNums = config.getMaxBlockNums();
    // TODO: how to alloc block nums more flexable according to shape
    constexpr int BlockPerBeam = 8;
    int maxBlockNums = 1024;
    int BlockNums1 = std::min(batch_size * beamwidth * BlockPerBeam, maxBlockNums);
    int BlockNums2 = std::min(batch_size, maxBlockNums);
    dim3 grid_round1(BlockNums1);
    dim3 block_round1(256);
    dim3 grid_round2(BlockNums2);
    dim3 block_round2(256);
    // debug info, better to retain: std::cout << "in cu file, before launch" << std::endl;
    topK_kernel_round1<beamwidth, 256, BlockPerBeam>
                        <<<grid_round1, block_round1>>>(probs, vocab_size, topK_ids, topK_vals);
    topK_kernel_round2<beamwidth, 256, BlockPerBeam>
                        <<<grid_round2, block_round2>>>(topK_ids, topK_vals, final_topK_ids, final_topK_vals);
    // debug info, better to retain: std::cout << "in cu file, after launch" << std::endl;
}
