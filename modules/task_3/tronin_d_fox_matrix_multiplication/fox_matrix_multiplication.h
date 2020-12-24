// Copyright 2020 Tronin Dmitry
#ifndef MODULES_TASK_3_TRONIN_D_FOX_MATRIX_MULTIPLICATION_FOX_MATRIX_MULTIPLICATION_H_
#define MODULES_TASK_3_TRONIN_D_FOX_MATRIX_MULTIPLICATION_FOX_MATRIX_MULTIPLICATION_H_

#include <vector>

std::vector<double> getRandomVector(size_t size);
std::vector<double> SequentialMultiplication(const std::vector<double>& lhs,
                                             const std::vector<double>& rhs,
                                             size_t n);

std::vector<double> ParallelFoxMultiplication(const std::vector<double>& lhs,
                                              const std::vector<double>& rhs,
                                              size_t n);

#endif  // MODULES_TASK_3_TRONIN_D_FOX_MATRIX_MULTIPLICATION_FOX_MATRIX_MULTIPLICATION_H_
