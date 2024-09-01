#pragma once
#include <vector>


void FilterByConfidenceCuda(int classesNumber, int vectorSize,
                        float transOutputVector[1][8400][84], 
                        std::vector<int> &indexes, 
                        std::vector<float> &class_ind, 
                        std::vector<float> &conf_vector, 
                        float confThreshold);