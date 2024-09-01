#include "getImgParameters.h"

/// @brief function for get params of img
/// @param cv::Mat& img,int* imgParams
/// @return void;
void getImgParameters(cv::Mat& img,int* imgParams) {
    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    imgParams[0] = width;
    imgParams[1] = height;
    imgParams[2] = channels;
}

