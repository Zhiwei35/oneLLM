#include <stdio.h>
#include "src/utils/model_utils.h"

struct Config {
	std::string dir = "../llamaweight/"; // 模型文件路径
    std::string tokenizer_file = "../llama2-7b-tokenizer.bin";
    int max_seq_len = -1; //输出句子的最大长度，超出即退出
};

int main(int argc, char **argv) {
    int round = 0;
    std::string history = "";

    Config model_path;

 	std::vector <std::string> sargv;
	for (int i = 0; i < argc; i++) {
		sargv.push_back(std::string(argv[i]));
	}   
	for (int i = 1; i < argc; i++) {
        if (sargv[i] == "-p" || sargv[i] == "--path") {
			model_path.dir = sargv[++i];
        }
        if (sargv[i] == "-t" || sargv[i] == "--tokenizer_path") {
			model_path.tokenizer_file = sargv[++i];
        }
        // placeholder for more args
    }
    // 加载模型到自定义的model data structure，这一块去看看ft的实现，我感觉那一块更好
    auto model = onellm::CreateOneLLMModelFromDummy<float>(model_path.tokenizer_file);//model.cpp拿到对应model class的pointer，同时load weight到该class的数据结构中
//    auto model = onellm::CreateOneLLMModelFromFile<half>(model_path.dir, model_path.tokenizer_file);//model.cpp拿到对应model class的pointer，同时load weight到该class的数据结构中
    std::string model_name = model->model_name;
    // exist when generate end token or reach max seq
    while (true) {
        printf("please input the question: ");
        std::string input;
        std::getline(std::cin, input);
        if (input == "reset") {// 清空当前所有内容和轮次，重新开始
            history = "";
            round = 0;
            continue;
        }
        if (input == "stop") {//停止对话
            break;
        }    
        //index可以认为生成的第几个token，从0开始
        std::string retString = model->Response(model->MakeInput(history, round, input), [model_name](int index, const char* content) {
            if (index == 0) {
                printf("%s:%s", model_name.c_str(), content);
                fflush(stdout);
            }
            if (index > 0) {
                printf("%s", content);
                fflush(stdout);
            }
            if (index == -1) {
                printf("\n");
            }
        });
        //多轮对话保留history，制作成新的上下文context
        history = model->MakeHistory(history, round, input, retString);
        round++;
    }
    return 0;
}
