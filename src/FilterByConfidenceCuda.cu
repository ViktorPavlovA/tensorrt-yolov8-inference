#include "FilterByConfidenceCuda.h"
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

// CUDA kernel
__global__ void filterByConfidenceKernel(int classesNumber, int vectorSize,
                                         float *transOutputVector,
                                         int *indexes,
                                         float *class_ind,
                                         float *conf_vector,
                                         float confThreshold, 
                                         int *outCount) {
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Calculate global index
    if (i < vectorSize) {
        float max_element = 0.0;
        float ind = -1.0; // Initialize to -1 to signify no class

        for (int j = 4; j < (classesNumber + 4); ++j) {
            float current_value = transOutputVector[i * 84 + j];
            if (max_element < current_value) { // Flatten the 2D array
                max_element = current_value;
                ind = static_cast<float>(j); // Save the index of the max class
            }
        }
        if (max_element > confThreshold) {
            int count = atomicAdd(outCount, 1); // Use atomic counter for output index
            indexes[count] = i; // Store index
            class_ind[count] = ind - 4.0f; // Store class index
            conf_vector[count] = max_element; // Store confidence
        }
    }
}

// Host function to initialize and call the CUDA kernel
void FilterByConfidenceCuda(int classesNumber, int vectorSize,
                             float transOutputVector[1][8400][84],
                             std::vector<int> &indexes,
                             std::vector<float> &class_ind,
                             std::vector<float> &conf_vector,
                             float confThreshold) {

    // Define sizes
    size_t transOutputVectorSize = vectorSize * 84 * sizeof(float);
    
    // Device pointers
    int *d_indexes;
    float *d_class_ind;
    float *d_conf_vector;
    float *d_transOutputVector;
    int *d_outCount;
    
    // Allocate device memory
    cudaMalloc(&d_transOutputVector, transOutputVectorSize);
    cudaMalloc(&d_indexes, vectorSize * sizeof(int));
    cudaMalloc(&d_class_ind, vectorSize * sizeof(float));
    cudaMalloc(&d_conf_vector, vectorSize * sizeof(float));
    cudaMalloc(&d_outCount, sizeof(int));
    
    // Initialize output counter on device
    int zero = 0;
    cudaMemcpy(d_outCount, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // Copy data to device
    cudaMemcpy(d_transOutputVector, transOutputVector, transOutputVectorSize, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;
    filterByConfidenceKernel<<<blocksPerGrid, threadsPerBlock>>>(classesNumber, vectorSize,
        d_transOutputVector, d_indexes, d_class_ind, d_conf_vector, confThreshold, d_outCount);
    
    // Copy results back to host
    int outCount;
    cudaMemcpy(&outCount, d_outCount, sizeof(int), cudaMemcpyDeviceToHost);
    
    indexes.resize(outCount);
    class_ind.resize(outCount);
    conf_vector.resize(outCount);
    
    cudaMemcpy(indexes.data(), d_indexes, outCount * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(class_ind.data(), d_class_ind, outCount * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(conf_vector.data(), d_conf_vector, outCount * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_transOutputVector);
    cudaFree(d_indexes);
    cudaFree(d_class_ind);
    cudaFree(d_conf_vector);
    cudaFree(d_outCount);
}