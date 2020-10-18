// Copyright 2020 Kochankov Ilya

#ifndef MODULES_TASK_1_KOCHANKOV_I_MATRIX_COLUMNS_SUM_MATRIX_COLUMNS_SUM_H_
#define MODULES_TASK_1_KOCHANKOV_I_MATRIX_COLUMNS_SUM_MATRIX_COLUMNS_SUM_H_

#include <vector>

using std::vector;

std::vector<vector<double >> get_rand_matrix(int x, int y);
double* sequential_operations(int result_size, int start_index, const vector<vector<double>>& matrix);
vector<double> parallel_sum(const vector<vector<double >>& matrix);
vector<double> not_parallel_sum(const vector<vector<double >>& matrix);

#endif  // MODULES_TASK_1_KOCHANKOV_I_MATRIX_COLUMNS_SUM_MATRIX_COLUMNS_SUM_H_
