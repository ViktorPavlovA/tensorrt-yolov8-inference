#pragma once    

#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>


/// @brief Classes of coco dataset
struct cocoObject {
 std::vector<std::pair<std::string, cv::Scalar>> vectorClassesColors = {
        {"person", cv::Scalar(220, 20, 60)},
        {"bicycle", cv::Scalar(119, 11, 32)},
        {"car", cv::Scalar(0, 0, 142)},
        {"motorcycle", cv::Scalar(0, 0, 230)},
        {"airplane", cv::Scalar(106, 0, 228)},
        {"bus", cv::Scalar(0, 60, 100)},
        {"train", cv::Scalar(0, 80, 100)},
        {"truck", cv::Scalar(0, 0, 192)},
        {"boat", cv::Scalar(153, 153, 153)},
        {"traffic light", cv::Scalar(250, 170, 30)},
        {"fire hydrant", cv::Scalar(250, 0, 30)},
        {"stop sign", cv::Scalar(0, 0, 255)},
        {"parking meter", cv::Scalar(0, 225, 255)},
        {"bench", cv::Scalar(0, 128, 0)},
        {"bird", cv::Scalar(0, 255, 255)},
        {"cat", cv::Scalar(255, 0, 0)},
        {"dog", cv::Scalar(255, 255, 0)},
        {"horse", cv::Scalar(0, 0, 128)},
        {"sheep", cv::Scalar(255, 0, 255)},
        {"cow", cv::Scalar(128, 0, 128)},
        {"elephant", cv::Scalar(255, 165, 0)},
        {"bear", cv::Scalar(0, 128, 128)},
        {"zebra", cv::Scalar(128, 128, 128)},
        {"giraffe", cv::Scalar(128, 255, 0)},
        {"backpack", cv::Scalar(255, 255, 255)},
        {"umbrella", cv::Scalar(0, 0, 0)},
        {"handbag", cv::Scalar(128, 0, 0)},
        {"tie", cv::Scalar(255, 128, 0)},
        {"suitcase", cv::Scalar(0, 128, 255)},
        {"frisbee", cv::Scalar(255, 255, 160)},
        {"skis", cv::Scalar(0, 255, 160)},
        {"snowboard", cv::Scalar(0, 160, 255)},
        {"sports ball", cv::Scalar(160, 0, 255)},
        {"kite", cv::Scalar(160, 160, 0)},
        {"baseball bat", cv::Scalar(0, 0, 160)},
        {"baseball glove", cv::Scalar(0, 160, 160)},
        {"skateboard", cv::Scalar(160, 160, 200)},
        {"surfboard", cv::Scalar(160, 200, 160)},
        {"tennis racket", cv::Scalar(200, 0, 0)},
        {"bottle", cv::Scalar(0, 200, 0)},
        {"wine glass", cv::Scalar(0, 0, 200)},
        {"cup", cv::Scalar(200, 200, 0)},
        {"fork", cv::Scalar(200, 255, 200)},
        {"knife", cv::Scalar(200, 0, 255)},
        {"spoon", cv::Scalar(0, 200, 255)},
        {"bowl", cv::Scalar(0, 200, 200)},
        {"banana", cv::Scalar(255, 255, 128)},
        {"apple", cv::Scalar(255, 0, 0)},
        {"sandwich", cv::Scalar(0, 255, 0)},
        {"orange", cv::Scalar(0, 128, 255)},
        {"broccoli", cv::Scalar(0, 255, 128)},
        {"carrot", cv::Scalar(128, 64, 0)},
        {"hot dog", cv::Scalar(128, 0, 64)},
        {"pizza", cv::Scalar(128, 128, 0)},
        {"donut", cv::Scalar(0, 255, 255)},
        {"cake", cv::Scalar(0, 128, 0)},
        {"chair", cv::Scalar(112, 128, 144)},
        {"couch", cv::Scalar(0, 0, 128)},
        {"potted plant", cv::Scalar(240, 128, 128)},
        {"bed", cv::Scalar(255, 192, 203)},
        {"dining table", cv::Scalar(255, 0, 128)},
        {"toilet", cv::Scalar(255, 165, 0)},
        {" TV", cv::Scalar(128, 255, 128)},
        {"laptop", cv::Scalar(255, 255, 0)},
        {"mouse", cv::Scalar(128, 0, 255)},
        {"remote", cv::Scalar(0, 255, 128)},
        {"keyboard", cv::Scalar(128, 128, 255)},
        {"cell phone", cv::Scalar(0, 255, 255)},
        {"microwave", cv::Scalar(128, 255, 0)},
        {"oven", cv::Scalar(255, 128, 128)},
        {"toaster", cv::Scalar(255, 255, 255)},
        {"sink", cv::Scalar(0, 0, 255)},
        {"refrigerator", cv::Scalar(0, 0, 128)},
        {"book", cv::Scalar(255, 0, 255)},
        {"clock", cv::Scalar(128, 0, 128)},
        {"vase", cv::Scalar(255, 165, 0)},
        {"scissors", cv::Scalar(0, 128, 255)},
        {"teddy bear", cv::Scalar(0, 128, 128)},
        {"hair drier", cv::Scalar(128, 128, 128)},
        {"toothbrush", cv::Scalar(255, 0, 128)}
    };
};
