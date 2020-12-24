  // Copyright 2020 Ivanov Yaroslav
#ifndef MODULES_TASK_3_IVANOV_Y_LINEFILTER_BLOCK_LINEFILTER_BLOCK_H_
#define MODULES_TASK_3_IVANOV_Y_LINEFILTER_BLOCK_LINEFILTER_BLOCK_H_

#include <vector>

std::vector<int> CreateRandomImg(const int width, const int height);
int LocalFilter(const std::vector<int>& Arr, int height, int width, int radius,
 float sigma, const std::vector<float>& Kernel, int row, int cols);
std::vector<int> ParallelFilter(const std::vector<int> &Arr, int height, int width, int radius,
 float sigma, const std::vector<float>& Kernel);
std::vector<int> ParallelFilterDeep(const std::vector<int>& Arr, int height, int width, int radius,
 float sigma, const std::vector<float>& Kernel, int begin, int add);
void Show(std::vector<int> lol, int height, int width);
std::vector<float> createKernel(int radius, float sigma);
int checkBoundaries(int data, int left, int right);
int createDiameter(int radius);

#endif  // MODULES_TASK_3_IVANOV_Y_LINEFILTER_BLOCK_LINEFILTER_BLOCK_H_
