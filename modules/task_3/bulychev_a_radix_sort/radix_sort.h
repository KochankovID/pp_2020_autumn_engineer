// Copyright 2020 Bulychev Andrey
#ifndef MODULES_TASK_3_BULYCHEV_A_RADIX_SORT_RADIX_SORT_H_
#define MODULES_TASK_3_BULYCHEV_A_RADIX_SORT_RADIX_SORT_H_

#include <mpi.h>

#include <string>
#include <vector>

std::vector<int> getRandomVector(int size);
int pow10(int power);
std::vector<int> bubble_sort_Vector(std::vector<int> vec);
int pow2(int pow);
bool vector_is_sort(std::vector<int> vec);
int get_max_power(std::vector<int> vec);
std::vector<int> RadixSort(std::vector<int> vec, int size);
std::vector<int> Merge(std::vector<int> vec1, std::vector<int> vec2);
std::vector<int> ParallSort(std::vector<int> vec);

#endif  // MODULES_TASK_3_BULYCHEV_A_RADIX_SORT_RADIX_SORT_H_
