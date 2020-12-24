// Copyright 2020 Egorov Kirill
#ifndef MODULES_TASK_3_EGOROV_K_BATCHER_SORT_BATCHER_SORT_H_
#define MODULES_TASK_3_EGOROV_K_BATCHER_SORT_BATCHER_SORT_H_

#include <vector>
#include <string>

std::vector<int> getRandomVector(int  sz);
void compEx(std::vector<int> *vec, int i, int j);
void sh(std::vector<int> *vec, int l, int r);
void uns(std::vector<int> *vec, int l, int r);

bool simpleCheck(std::vector<int> const &vec);
void simpleSort(std::vector<int> *vec);

void print(std::vector <int> const &a);

void BatcherSort(std::vector<int> *vec);
void BatcherSortParallel(std::vector<int> *vec);

#endif  // MODULES_TASK_3_EGOROV_K_BATCHER_SORT_BATCHER_SORT_H_
