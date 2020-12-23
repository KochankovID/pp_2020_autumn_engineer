
// Copyright 2020 Shchelyanskova Nastya
#ifndef MODULES_TASK_3_SHCHELYANSKOVA_N_RADIX_SORT_DOUBLE_SIMPLE_MERGE_RADIX_SORT_DOUBLE_SIMPLE_MERGE_H_
#define MODULES_TASK_3_SHCHELYANSKOVA_N_RADIX_SORT_DOUBLE_SIMPLE_MERGE_RADIX_SORT_DOUBLE_SIMPLE_MERGE_H_
#include <vector>

// Получение случайного вектора
std::vector<double> getRandomVector(int size);
// Сортирует массив сортировкой пузырьком
std::vector<double> bubbleSort(const std::vector<double>& vec);
// Получение кол-ва разрядов в числе (положительных)
int numberOfPositiveRadix(int num);
// Получение кол-ва разрядов в числе (положительных)
int numberOfNegativeRadix(double num);
// Получение цифры в указанном разряде из числа
int getDigit(double num, int discharge);
// Слияние упорядоченных массивов
std::vector<double> merge(const std::vector<double>& vec1, const std::vector<double>& vec2);
// Линейная поразрядная сортировка
std::vector<double> linearRadixSort(const std::vector<double>& vect);
// Сортировка по определенному разряду
std::vector<double> radixSort(const std::vector<double>& vect, int radix);
// Паралельная поразрядная сортировка
std::vector<double> parallelRadixSort(const std::vector<double>& vect);

#endif  // MODULES_TASK_3_SHCHELYANSKOVA_N_RADIX_SORT_DOUBLE_SIMPLE_MERGE_RADIX_SORT_DOUBLE_SIMPLE_MERGE_H_
