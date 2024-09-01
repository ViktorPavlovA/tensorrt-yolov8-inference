#include "getRatioWH.h"



/// @brief function for get ratio of img;
/// @param int* imgParams,float* ratio, int inputWidth, int inputHeight;
/// @return void;
void getRatioWH(int* imgParams, float* ratio,int inputWidth, int inputHeight) {
    ratio[0] = (float)imgParams[0] /(float)inputWidth;
    ratio[1] = (float)imgParams[1] /(float)inputHeight;
}