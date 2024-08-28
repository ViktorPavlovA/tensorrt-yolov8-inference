#include "logger.h"


/*
Declaration method "log" of class;
*/
void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
};


