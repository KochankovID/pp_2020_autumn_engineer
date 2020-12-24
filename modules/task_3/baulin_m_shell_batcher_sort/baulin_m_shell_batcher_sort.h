// Copyright 2020 Mikhail Baulin
#ifndef MODULES_TASK_3_BAULIN_M_SHELL_BATCHER_SORT_BAULIN_M_SHELL_BATCHER_SORT_H_
#define MODULES_TASK_3_BAULIN_M_SHELL_BATCHER_SORT_BAULIN_M_SHELL_BATCHER_SORT_H_

#include <vector>

void S(std::vector<int> procs_up, std::vector<int> procs_down);
void B(std::vector<int> procs);

std::vector<int> getVectorWithRandomNumbers(int size, int limit);
void getSequentialOperations(std::vector<int>* vec);
std::vector<int> getParallelOperations(std::vector<int> vec);

#endif  // MODULES_TASK_3_BAULIN_M_SHELL_BATCHER_SORT_BAULIN_M_SHELL_BATCHER_SORT_H_
