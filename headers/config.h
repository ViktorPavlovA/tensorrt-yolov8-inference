/*
USED NETRON WEB APP TO FIND YOU INPUT-OUTPUT SHAPE OF MODEL 

https://netron.app/
*/
#include <string>

/// @brief Parametrs:
///
/// int inputWidth - input in onnx viewer; 
///
/// int inputHeight - input in onnx viewer;
///
/// int batchSize - batch inference;
///
/// int classesNumber - output vector 1x84x8400 -  x_c,y_c, w, h + 80 classes in coco dataset;
///
/// int vectorSize - 8400 look in to onnx model;
///
/// int inputSize - inputHeight * inputWidth * 3 input in onnx viewer;
///
/// int outputSize - 1 * (classesNumber+4) * vectorSize output vector look in onnx viewer
///
/// float iou_threshold - iou  threshold for nms;
///
/// float conf - confidence threshold for nms;
///
/// std::string modelPath - way to model;
///
/// int sourceCam - source camera;
///
/// std::string sourceVideo - example used videosource;
struct ModelParameters {
    int inputWidth = 640; // input in onnx viewer
    int inputHeight = 640; // input in onnx viewer
    int batchSize = 1;
    int classesNumber = 80; // output vector 1x84x8400 -  x_c,y_c, w, h + 80 classes in coco dataset;
    int vectorSize = 8400; 
    int inputSize = inputHeight * inputWidth * 3; // input in onnx viewer
    int outputSize = 1 * (classesNumber+4) * vectorSize; // output vector look in onnx viewer
    float iou_threshold = 0.4f; 
    float conf = 0.5f; // YOU CAN CHANGE THIS PARAMETER AS YOU WISH
    std::string modelPath = "../models/yolov8n.engine"; // way to model
    int sourceCam = 0;
    std::string sourceVideo = "../videos/video.mp4"; // example
};