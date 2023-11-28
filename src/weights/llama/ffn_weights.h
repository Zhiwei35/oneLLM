#pragma once
#include "src/weights/base_weights.h"

struct LLaMAFFNWeights {
    BaseWeight gate;
    BaseWeight up;
    BaseWeight down;
};
