#pragma once
#include "src/weights/base_weights.h"

struct LLaMAattentionWeights {
    BaseWeight q;
    BaseWeight k;
    BaseWeight v;
    BaseWeight qkv;
    BaseWeight output;
};
