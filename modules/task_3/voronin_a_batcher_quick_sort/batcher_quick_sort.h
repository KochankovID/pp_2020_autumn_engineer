// Copyright 2020 Voronin Aleksey
#ifndef MODULES_TASK_3_VORONIN_A_BATCHER_QUICK_SORT_BATCHER_QUICK_SORT_H_
#define MODULES_TASK_3_VORONIN_A_BATCHER_QUICK_SORT_BATCHER_QUICK_SORT_H_

#include <vector>

std::vector<int> getRandomVector(int size);
std::vector<int> quickSort(std::vector<int> vector);
void quickSort(int *vector, int left, int right);
void net(std::vector<int> vector);
void makeNetwork(int size);
void oddEvenMerge(std::vector<int> left, std::vector<int> right);
std::vector<int> parallelSort(std::vector<int> vec);

#endif  // MODULES_TASK_3_VORONIN_A_BATCHER_QUICK_SORT_BATCHER_QUICK_SORT_H_
