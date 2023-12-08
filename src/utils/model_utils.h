#include "src/models/basemodel.h"
#include "src/models/llama/llama.h"
#include "src/utils/macro.h"

namespace onellm {
    BaseModel *CreateModelWithName(const std::string& model_name) {
        ONELLM_CHECK_WITH_INFO(model_name == "llama", "dont support other models except llama yet!") 
        BaseModel *model = new Llama();
        return model;
    }
    // std::unique_ptr<baseModel> CreateOneLLMModelFromFile(std::string model_path){
    //     baseModel *model = CreateModelWithName("llama");
    //     model->loadWeights(fileName);
    //     // model->WarmUp();
    //     return std::unique_ptr<baseModel> (model);        
    // }
    std::unique_ptr<BaseModel> CreateOneLLMModelFromDummy(){
        BaseModel *model = CreateModelWithName("llama");
        model->loadWeightsFromDummy();
        // model->WarmUp();
        return std::unique_ptr<BaseModel> (model);        
    }
}