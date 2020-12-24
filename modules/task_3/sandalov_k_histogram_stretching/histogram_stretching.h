// Copyright 2020 Konstantin Sandalov
#ifndef MODULES_TASK_3_SANDALOV_K_HISTOGRAM_STRETCHING_HISTOGRAM_STRETCHING_H_
#define MODULES_TASK_3_SANDALOV_K_HISTOGRAM_STRETCHING_HISTOGRAM_STRETCHING_H_

#include <vector>
#include <random>

void createRandomImage(std::vector<int>* randImg, int width = 100, int height = 100);

int histogramStretching(std::vector<int>* targetImg = nullptr, int minVal = 0, int maxVal = 255);

int countMinMax(std::vector<int>* targetImg = nullptr, int* minVal = nullptr, int* maxVal = nullptr);

int seqHistogramStretching(std::vector<int>* targetImg = nullptr);

int parallelHistogramStretching(std::vector<int>* targetImg = nullptr);

#endif  // MODULES_TASK_3_SANDALOV_K_HISTOGRAM_STRETCHING_HISTOGRAM_STRETCHING_H_
