// Copyright 2020 Kryukov Sergey
#ifndef MODULES_TASK_3_KRYUKOV_S_OPERSOBEL_OPERSOBEL_H_
#define MODULES_TASK_3_KRYUKOV_S_OPERSOBEL_OPERSOBEL_H_

#include <mpi.h>
#include <vector>
#include <string>

std::vector<int> sobel_for_oneproc(std::vector<int> img, int row, int col);
std::vector<int> paral_sobel_oper(std::vector<int> img, int row, int col);
int correctRange(int value, int max, int min);
int use_sobel_kernel(std::vector<int> img, int x, int y, int row, int col);


#endif  // MODULES_TASK_3_KRYUKOV_S_OPERSOBEL_OPERSOBEL_H_
