
/*
Definition of supply methods
*/
#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <cassert>
#include <algorithm>
#include <NvInfer.h>

/// @brief function for get params of img
/// @param cv::Mat& img - img
/// @return std::vector<int> - [w,h,channels];
std::vector<int> getImgParameters(cv::Mat& img);


/// @brief function for get ratio of img;
/// @param std::vector<int>& imgParams - vector param of img
/// @return std::vector<float> - [ratioW, ratioH];
std::vector<float> getRatioWH(std::vector<int>& imgParams, int inputWidth, int inputHeight);



/// @brief function for preprocessing image;
/// @param cv::Mat& img_original - original img, int inputWidth, int inputHeight , float* input_vector
/// @return void;
void preprocessImage(cv::Mat& img_original, int inputWidth, int inputHeight, float* input_vector);


/// @brief function for inference;
/// @param 
/// IExecutionContext* context - current context,
/// void* buffers[] - buffer params, 
/// int batchSize - batch count,
/// @return void;
void performInference(nvinfer1::IExecutionContext* context, void* buffers[], int batchSize);


/// @brief function for upload model;
/// @param 
/// IExecutionContext* context - * on context,
/// void* buffers[] - buffer param, 
/// int batchSize - batch count,
/// @return nvinfer1::ICudaEngine* engine * on model;
nvinfer1::ICudaEngine* loadEngine(const std::string &engineFile, nvinfer1::ILogger &logger);



/// @brief supply function for calculating IoU;
float calculateIoU(std::vector<float>& box1,std::vector<float>& box2);

/// @brief fuction for filtering by IoU;
/// @param 
///
/// std::vector<std::vector<float>>& finalVector - sorted vector [x,y, x2,y2,conf,class];
/// 
/// float iou_threshold - threshold IoU;
/// @return void;
void filterByIoU(std::vector<std::vector<float>>& finalVector, float iou_threshold);

/// @brief function for transposing output vector;
/// @param input int classesNumber, int vectorSize, auto outputVector, auto transOutputVector; 
/// @return void;
void transposeOutputVector(int classesNumber, int vectorSize, float outputVector[1][84][8400], float transOutputVector[1][8400][84]);


/// @brief function for filtering by confidence;
/// @param input int classesNumber, int vectorSize, auto outputVector, auto transOutputVector; auto& indexes, auto& class_ind, auto& conf, float confThreshold;
/// @return void;
void FilterByConfidence(int classesNumber, int vectorSize,float transOutputVector[1][8400][84], std::vector<int>& indexes, std::vector<float>& class_ind, std::vector<float>& conf_vector, float confThreshold);


/// @brief - function for get final vector: x1,y1,x2,y2,conf,class_value;
/// @param auto& finalVector,auto transOutputVector, auto& indexes, auto& ratio, auto& class_ind, auto& conf_vector ; 
/// @return void;
void finalVectorMaker(std::vector<std::vector<float>>& finalVector,float transOutputVector[1][8400][84], std::vector<int>& indexes, std::vector<float>& ratio, std::vector<float>& class_ind, std::vector<float>& conf_vector);