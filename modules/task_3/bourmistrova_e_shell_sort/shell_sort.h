// Copyright 2020 Ekaterina Burmistrova
#ifndef MODULES_TASK_3_BOURMISTROVA_E_SHELL_SORT_SHELL_SORT_H_
#define MODULES_TASK_3_BOURMISTROVA_E_SHELL_SORT_SHELL_SORT_H_

#include <vector>

std::vector<int> gen_input(int  sz);
std::vector<int> Sequential_Shell(std::vector<int> vec);
std::vector<int>Sequential_sort(std::vector<int> vec);
std::vector<int> Parallel_sort(std::vector<int> vect);

#endif  // MODULES_TASK_3_BOURMISTROVA_E_SHELL_SORT_SHELL_SORT_H_
