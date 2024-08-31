#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <algorithm>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <chrono>

#include "headers/logger.h"
#include "headers/config.h"
#include "headers/utils.h"
#include "headers/coco.h"


int main(){
    Logger logger;
    ModelParameters parameters;
    cocoObject colorParam;
    const std::string enginePath = parameters.modelPath;
    cv::Mat inputImg;
    // Init parameters
    std::unique_ptr<float[]> input_vector(new float[parameters.inputWidth * parameters.inputHeight * 3]);
    float* d_input;
    float* d_output;
    int inputSize = parameters.inputSize; // Assuming 3 channels (RGB)
    int outputSize = parameters.outputSize; // model output
    cudaMalloc((void**) &d_input, inputSize * sizeof(float));    
    cudaMalloc((void**) &d_output, outputSize * sizeof(float));
    std::cout << "Alocated space for params" << std::endl;
    void* buffers[2];
    buffers[0] = d_input;   // Input buffer
    buffers[1] = d_output;  // Output buffer


    auto timeStart = std::chrono::high_resolution_clock::now();
    auto timeEnd = std::chrono::high_resolution_clock::now();
    double fps = 0.0;
    std::chrono::duration<double> duration ;

    // CHANGE THIS PLACE
    float outputVector[1][84][8400];
    float transOutputVector[1][8400][84];

    
    std::vector<int> indexes;
    std::vector<float> class_ind;
    std::vector<float> conf_vector;
    std::vector<std::vector<float>> finalVector;

    // init class cap & don't foget choose source
    cv::VideoCapture cap(parameters.sourceCam);

    // Get first frame
    if (!cap.isOpened()) {
        std::cerr << "ERROR! Unable to open camera\n";
        return -1;
    }
    while (true){
        cap.read(inputImg);
        if (inputImg.empty()) {
            std::cerr << "ERROR! blank frame grabbed\n";
        }
        else {
            std::cout << "Frame captured successfully!" << std::endl;
            break;
        }
    }
    std::vector<int> imgParams = getImgParameters(inputImg);
    std::vector<float> ratio = getRatioWH(imgParams,parameters.inputWidth,parameters.inputHeight);

    nvinfer1::ICudaEngine *cudaEngine = loadEngine(enginePath,logger);
    if (cudaEngine == nullptr){
        std::cout << "Error: Failed to load engine" << std::endl;
        return -1;
    }

    std::cout << "Creating execution context..." << std::endl;
    nvinfer1::IExecutionContext *context = cudaEngine->createExecutionContext();
    if (!context) {
        std::cerr << "Failed to create execution context!" << std::endl;
        // cudaEngine->destroy(); // Clean up the engine if context creation fails
        return -1; // Exit with error
    }
    std::cout << "Execution context created successfully!" << std::endl;


    while (true) {
        indexes.clear();
        class_ind.clear();
        conf_vector.clear();
        finalVector.clear();
        cap.read(inputImg);
        if (!inputImg.empty()) {
            preprocessImage(inputImg, parameters.inputWidth, parameters.inputHeight,input_vector.get());

            cudaMemcpy(d_input, input_vector.get(), inputSize * sizeof(float), cudaMemcpyHostToDevice);
            std::cout << "Copy img to cuda" << std::endl;
            performInference(context, buffers, 1); // Batch size of 1
            std::cout << "Make inference" << std::endl;
            cudaMemcpy(outputVector, d_output, outputSize * sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << "Copy output to host" << std::endl;

            transposeOutputVector(parameters.classesNumber,parameters.vectorSize,outputVector, transOutputVector);
            

            FilterByConfidence(parameters.classesNumber, parameters.vectorSize,transOutputVector, indexes, class_ind, conf_vector, parameters.conf);

            finalVectorMaker(finalVector,transOutputVector, indexes, ratio, class_ind, conf_vector);

            filterByIoU(finalVector, parameters.iou_threshold);
            
            for (const auto& row : finalVector) {
                cv::putText(inputImg, colorParam.vectorClassesColors[static_cast<int>(row[5])].first.c_str(), cv::Point(row[0], row[1]-20), cv::FONT_HERSHEY_SIMPLEX, 1, colorParam.vectorClassesColors[static_cast<int>(row[5])].second, 3);

                cv::rectangle(inputImg, cv::Point(row[0], row[1]), cv::Point(row[2], row[3]), colorParam.vectorClassesColors[static_cast<int>(row[5])].second, 2);
            }

            timeEnd = std::chrono::high_resolution_clock::now();
            duration = timeEnd - timeStart;
            fps = 1.0/duration.count();
            std::cout << "FPS: " << fps << std::endl;

            cv::putText(inputImg,std::to_string(fps) , cv::Point((imgParams[0]/2), (imgParams[1]/2)), cv::FONT_HERSHEY_SIMPLEX, 1, (255,255,0), 4);
            cv::imshow("Result", inputImg);
            int  key = cv::waitKey(2);
            if (key == 'q'){
                break;
            }    
        }
        else {
            std::cout << "No image for analyze" << std::endl;
        }
        timeStart = timeEnd;

    }
    // context->destroy(); // Destroy the context
    // cudaEngine->destroy(); // Destroy the engine
    cudaFree(d_input); // free the input buffer
    cudaFree(d_output);// free the output buffer
    std::cout << "Cleanup done." << std::endl;
    return 0;
}
