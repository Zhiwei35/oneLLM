#include <string>
#include <fstream>
#include <iostream>

[[noreturn]] inline void throwRuntimeError(const char* const file, int const line, std::string const& info = "")
{
    throw std::runtime_error(std::string("[oneLLM][ERROR] ") + info + " Assertion fail: " + file + ":"
                             + std::to_string(line) + " \n");
}

inline void onellmAssert(bool result, const char* const file, int const line, std::string const& info = "")
{
    if (!result) {
        throwRuntimeError(file, line, info);
    }
}

#define ONELLM_CHECK(val) myAssert(val, __FILE__, __LINE__)
#define ONELLM_CHECK_WITH_INFO(val, info)                                                                                  \
    do {                                                                                                               \
        bool is_valid_val = (val);                                                                                     \
        if (!is_valid_val) {                                                                                           \
            onellmAssert(is_valid_val, __FILE__, __LINE__, (info));                                             \
        }                                                                                                              \
    } while (0)