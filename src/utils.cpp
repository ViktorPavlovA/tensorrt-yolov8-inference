#include "utils.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <string>
#include <cassert>
#include <algorithm>
#include "NvInfer.h"
#include <cuda_runtime_api.h>


/// @brief function for preprocessing image;
/// @param cv::Mat& img_original - original img, int inputWidth, int inputHeight , float* input_vector
/// @return void;
void preprocessImage(cv::Mat& img_original, int inputWidth, int inputHeight, float* input_vector) {
    cv::Mat img;
    cv::resize(img_original, img, cv::Size(inputWidth, inputHeight));
    const int channels = img.channels();
    const int width = img.cols;
    const int height = img.rows;

    for (int c = 0; c < channels; c++) {
        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                input_vector[c * width * height + h * width + w] =
                    img.at<cv::Vec3b>(h, w)[c] / 255.0f;
            }
        }
    }
}


/// @brief function for inference;
/// @param 
/// IExecutionContext* context - current context,
/// void* buffers[] - buffer params, 
/// int batchSize - batch count,
/// @return void;
void performInference(nvinfer1::IExecutionContext* context, void* buffers[], int batchSize) {
    // Launch the inference
    // context->execute(batchSize, buffers);
    context->executeV2(buffers);
}


/// @brief function for upload model;
/// @param 
/// IExecutionContext* context - * on context,
/// void* buffers[] - buffer param, 
/// int batchSize - batch count,
/// @return nvinfer1::ICudaEngine* engine * on model;
nvinfer1::ICudaEngine* loadEngine(const std::string &engineFile, nvinfer1::ILogger &logger) {
    std::ifstream file(engineFile, std::ios::binary);
    if (!file.good()) {
        return nullptr;
    }

    // Get the size of the serialized engine
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::cout << "Engine size: " << size << " bytes" << std::endl;

    // If the size is zero, return nullptr
    if (size < 100) {
        std::cerr << "Engine file is empty!" << std::endl;
        return nullptr;
    }

    // Allocate memory to hold the engine
    char *serializedEngine = new char[size];
    file.read(serializedEngine, size);
    file.close();

    // Create a TensorRT runtime and deserialize the engine
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);
    // nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(serializedEngine, size, nullptr);
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(serializedEngine, size);
    // Clean up the serialized engine after deserialization
    delete[] serializedEngine;

    if (!engine) {
        std::cerr << "Failed to create the engine!" << std::endl;
        delete runtime; // Ensure runtime is cleaned up
        return nullptr;
    }

    return engine;
}



/// @brief supply function for calculating IoU;
float calculateIoU(std::vector<float>& box1,std::vector<float>& box2) {
    // box1 и box2 имеют формат {x_c, y_c, w, h}

    // Вычисляем координаты углов ограничивающего прямоугольника
    float x1_min = box1[0] - (box1[2] / 2);
    float y1_min = box1[1] - (box1[3] / 2);
    float x1_max = box1[0] + (box1[2] / 2);
    float y1_max = box1[1] + (box1[3] / 2);

    float x2_min = box2[0] - (box2[2] / 2);
    float y2_min = box2[1] - (box2[3] / 2);
    float x2_max = box2[0] + (box2[2] / 2);
    float y2_max = box2[1] + (box2[3] / 2);

    // calculate intersection coordinates
    float inter_x_min = std::max(x1_min, x2_min);
    float inter_y_min = std::max(y1_min, y2_min);
    float inter_x_max = std::min(x1_max, x2_max);
    float inter_y_max = std::min(y1_max, y2_max);

    // Get the width and height of the intersection and area
    float inter_width = std::max(0.0f, inter_x_max - inter_x_min);
    float inter_height = std::max(0.0f, inter_y_max - inter_y_min);
    float area_of_intersection = inter_width * inter_height;

    // calculate the area of both boxes
    float area1 = box1[2] * box1[3]; // width * height
    float area2 = box2[2] * box2[3];

    // calculate the area of union
    float area_of_union = area1 + area2 - area_of_intersection;

    return area_of_intersection / area_of_union; // IoU
}


/// @brief fuction for filtering by IoU;
/// @param 
///
/// std::vector<std::vector<float>>& finalVector - sorted vector [x,y, x2,y2,conf,class];
/// 
/// float iou_threshold - threshold IoU;
/// @return void;
void filterByIoU(std::vector<std::vector<float>>& finalVector, float iou_threshold) {
    std::vector<std::vector<float>> filteredVector;

    for (size_t i = 0; i < finalVector.size(); ++i) {
        bool keep = true;
        for (size_t j = 0; j < filteredVector.size(); ++j) {
            if (finalVector[i][5] == filteredVector[j][5]) { // Сравниваем class_value
                float iou = calculateIoU(finalVector[i], filteredVector[j]);
                if (iou > iou_threshold) {
                    keep = false; // Удалить, если IoU превышает порог
                    break;
                }
            }
        }
        if (keep) {
            filteredVector.push_back(finalVector[i]);
        }
    }

    finalVector = filteredVector; // save filtred vector
}

/// @brief function for transposing output vector;
/// @param int classesNumber, int vectorSize, auto outputVector, auto transOutputVector; 
/// @return void;
void transposeOutputVector(int classesNumber, int vectorSize, float outputVector[1][84][8400], float transOutputVector[1][8400][84]) {
  for (int j = 0; j < (classesNumber+4); ++j) {
    for (int i = 0; i < vectorSize; ++i) {
      transOutputVector[0][i][j] = outputVector[0][j][i];
    }
  }
  std::cout << "Transpose output - done" << std::endl;
}



/// @brief function for filtering by confidence;
/// @param input int classesNumber, int vectorSize, auto outputVector, auto transOutputVector; auto& indexes, auto& class_ind, auto& conf, float confThreshold;
/// @return void;
void FilterByConfidence(int classesNumber, int vectorSize,float transOutputVector[1][8400][84], std::vector<int>& indexes, std::vector<float>& class_ind, std::vector<float>& conf_vector, float confThreshold){
    for (int i = 0; i < vectorSize ; ++i) {
        float max_element = 0.0;
        float ind = 0.0;

        for (int j = 4; j < (classesNumber+4); ++j) {
            if (max_element < transOutputVector[0][i][j]){
                max_element = transOutputVector[0][i][j];
                ind = static_cast<float>(j);
            }
        }
        if (max_element > confThreshold){
            indexes.push_back(i*1.0);
            class_ind.push_back(ind-4.0);
            conf_vector.push_back(max_element);
        }
    }
}


/// @brief - function for get final vector: x1,y1,x2,y2,conf,class_value;
/// @param auto& finalVector,auto transOutputVector, auto& indexes, auto& ratio, auto& class_ind, auto& conf_vector ; 
/// @return void;
void finalVectorMaker(std::vector<std::vector<float>>& finalVector,float transOutputVector[1][8400][84], std::vector<int>& indexes, float* ratio, std::vector<float>& class_ind, std::vector<float>& conf_vector){
    for (int i = 0; i < indexes.size(); ++i) {
        // init vector 
        finalVector.push_back(std::vector<float>(6));

        finalVector[i][0] = (transOutputVector[0][indexes[i]][0] - transOutputVector[0][indexes[i]][2]/2) * ratio[0]; //x1
        finalVector[i][1] = (transOutputVector[0][indexes[i]][1] - transOutputVector[0][indexes[i]][3]/2) * ratio[1]; //y1
        finalVector[i][2] = (transOutputVector[0][indexes[i]][0] + transOutputVector[0][indexes[i]][2]/2) * ratio[0];; //x2
        finalVector[i][3] = (transOutputVector[0][indexes[i]][1] + transOutputVector[0][indexes[i]][3]/2) * ratio[1]; //y2
        finalVector[i][4] = conf_vector[i]; //conf
        finalVector[i][5] = class_ind[i]; //class_value
    }
}