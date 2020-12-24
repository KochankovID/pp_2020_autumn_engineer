// Copyright 2020 Kren Polina
#ifndef MODULES_TASK_3_KREN_P_GAUSSIAN_KERNEL_GAUSSIAN_KERNEL_H_
#define MODULES_TASK_3_KREN_P_GAUSSIAN_KERNEL_GAUSSIAN_KERNEL_H_

#include <vector>

std::vector <int> get_Img(int cols, int rows);
int compute_pixel(const std::vector<int> mtx, const std::vector<int> mask, int i, int j, int rows, int cols);
std::vector <int> parall_linear_filter(const std::vector <int> mask, const std::vector <int> a, int rows, int cols);
std::vector <int> SequentialLinearFilter(const std::vector <int> mask, const std::vector <int> a, int rows, int cols);

#endif  // MODULES_TASK_3_KREN_P_GAUSSIAN_KERNEL_GAUSSIAN_KERNEL_H_
