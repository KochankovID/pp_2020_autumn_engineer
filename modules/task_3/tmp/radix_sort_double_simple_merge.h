
#ifndef MODULES_TASK_3_SHCHELYANSKOVA_N_RADIX_SORT_DOUBLE_SIMPLE_MERGE_H_
#define MODULES_TASK_3_SHCHELYANSKOVA_N_RADIX_SORT_DOUBLE_SIMPLE_MERGE_H_
#include <vector>

std::vector<double> getRandomVector(int size);  // Получение случайного вектора
std::vector<double> bubbleSort(const std::vector<double>& vec);  // Сортирует массив сортировкой пузырьком
int numberOfPositiveRadix(int num);  // Получение кол-ва разрядов в числе (положительных)
int numberOfNegativeRadix(double num);  // Получение кол-ва разрядов в числе (положительных)
int getDigit(double num, int discharge);  // Получение цифры в указанном разряде из числа
std::vector<double> merge(const std::vector<double>& vec1, const std::vector<double>& vec2);  // Слияние упорядоченных массивов
std::vector<double> linearRadixSort(const std::vector<double>& vect);  // Линейная поразрядная сортировка
std::vector<double> radixSort(const std::vector<double>& vect, int radix);  // Сортировка по определенному разряду
std::vector<double> parallelRadixSort(const std::vector<double>& vect);  // Паралельная поразрядная сортировка

#endif  // MODULES_TASK_3_SHCHELYANSKOVA_N_RADIX_SORT_DOUBLE_SIMPLE_MERGE_H_
