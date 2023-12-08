#include "src/models/basemodel.h"
#include "src/models/llama/llama.h"
#include "src/utils/macro.h"

namespace onellm {
    template<typename T>
    BaseModel *CreateModelWithName(const std::string& model_name) {
        ONELLM_CHECK_WITH_INFO(model_name == "llama", "dont support other models except llama yet!");
        BaseModel *model = new Llama<T>();
        return model;
    }

    template<typename T>
    std::unique_ptr<BaseModel> CreateOneLLMModelFromDummy(){
        BaseModel *model = CreateModelWithName<T>("llama");
        model->loadWeightsFromDummy();
        // model->WarmUp();
        return std::unique_ptr<BaseModel> (model);        
    }
}