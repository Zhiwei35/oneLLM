#include <cuda_runtime.h>
#include <float.h>
template<int K>
struct topK
{
    float val[K];
    int id[K];

    __device__ void init(){
        for (int i = 0; i < K; i++) {
            id[i] = -1;
            val[i] = FLT_MIN;
        }
    }

    __device__ void insertHeap(float data, int data_id){
        if(id[K-1] == -1 || val[K-1] < data){
            id[K-1] = data_id;
            val[K-1] = data;
        }
        //一轮冒泡排序，重排，只需一轮，因为此时除了最后一个元素，其它都是有序
        for (int i = K - 2; i >= 0; i--){
            if(val[i + 1] > val[i] || id[i] == -1) {
                float tmp = val[i];
                val[i] = val[i + 1];
                val[i + 1] = tmp;                
                int tmp_id = id[i];
                id[i] = id[i + 1];
                id[i + 1] = tmp_id;
            }
        }
    }
};

void launchTopKforBeamSearch(const float* probs, 
                            const int batch_size,
                            const int vocab_size, 
//                            const int beamwidth,
                            float* topk_workspace);
