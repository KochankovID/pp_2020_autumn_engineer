// Copyright 2020 Nasedkin Nikita
#ifndef MODULES_TASK_3_NASEDKIN_MOORE_ALGORITHM_MOORE_ALGOTIHTM_H_
#define MODULES_TASK_3_NASEDKIN_MOORE_ALGORITHM_MOORE_ALGOTIHTM_H_

#include <vector>

std::vector<int> GetRandomGraph(int size);
std::vector<int> MooreSequential(std::vector<int>* graph, int start);
std::vector<int> MooreParallel(std::vector<int>* graph, int start);

#endif  // MODULES_TASK_3_NASEDKIN_MOORE_ALGORITHM_MOORE_ALGOTIHTM_H_
