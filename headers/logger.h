/*
Definition of logger class
*/
#pragma once

#include "NvInfer.h"
#include <iostream>

/// @brief Class to make logger
class Logger : public nvinfer1::ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override;
};
