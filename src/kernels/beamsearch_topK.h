#include <cuda_runtime.h>
#include <float.h>
#include <cuda.h>
#include <cuda_fp16.h>

template<typename T, int K>
struct topK
{
    T val[K];
    int id[K];

    __device__ void init(){
        for (int i = 0; i < K; i++) {
            id[i] = -1;
            val[i] = -1e10;
        }
    }

    __device__ void insertHeap(T data, int data_id){
        if(id[K-1] == -1 || val[K-1] < data){
            id[K-1] = data_id;
            val[K-1] = data;
        }
        //Note: 仅需一轮冒泡排序（插入新元素的重排），因为此时除了最后一个新元素，其它都是有序
        for (int i = K - 2; i >= 0; i--){
            // TODO: not sure if I can directly run fp16 comparison
            if(val[i + 1] > val[i] || id[i] == -1) {
                T tmp = val[i];
                val[i] = val[i + 1];
                val[i + 1] = tmp;                
                int tmp_id = id[i];
                id[i] = id[i + 1];
                id[i + 1] = tmp_id;
            }
        }
    }
};

template<typename T>
void launchTopKforBeamSearch(TensorWrapper<T>* probs, 
                            TensorWrapper<T>* topk_workspace);
