
cmake_minimum_required(VERSION 3.10)
project(Yolov8)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(headers)

find_package(OpenCV REQUIRED)

#if you don't used jetson nano for PC 
include_directories(
    /usr/include/x86_64-linux-gnu
    /usr/local/cuda/include
)

#if you used jetson nano 
# include_directories(
#    /usr/lib/aarch64-linux-gnu
#    /usr/local/cuda/include
#)


link_directories(/usr/local/cuda/lib64)

add_executable(main main.cpp src/logger.cpp src/utils.cpp)

target_link_libraries(main nvinfer cudart ${OpenCV_LIBS})
