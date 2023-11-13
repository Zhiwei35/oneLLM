#include <algorithm>   // std::fill_n
#include <iostream>    // snprintf
#include <math.h>      // expf, log
#include <stdlib.h>    // rand
#include <string>      // std::string
#include <vector>      // std::vector

#include <cuda.h>

int main() {
    const int batch_size = 1;
    const int sequeue_length = 1024;
    const int vocab_size = 30000;
    // debug info, better to retain: std::cout <<"batch_size=" << batch_size << " sequeue_lenght=" << sequeue_length << "  vocab_size=" << vocab_size << std::endl;
}
