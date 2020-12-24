// Copyright 2020 Osipov Nikolay
#ifndef MODULES_TASK_2_OSIPOV_N_TOPOLOGY_RULER_TOPOLOGY_RULER_H_
#define MODULES_TASK_2_OSIPOV_N_TOPOLOGY_RULER_TOPOLOGY_RULER_H_

#include <vector>
#include <string>

std::vector<int> getVector(int  size);
int getParallelOperations(std::vector<int> global_vector,
    int count_size_vector, std::string operation);
int getSequentialOperations(std::vector<int> vector, std::string operation);


#endif  // MODULES_TASK_2_OSIPOV_N_TOPOLOGY_RULER_TOPOLOGY_RULER_H_
