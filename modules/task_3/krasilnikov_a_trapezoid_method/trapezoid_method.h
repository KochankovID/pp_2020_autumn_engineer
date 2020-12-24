// Copyright 2020 Krasilnikov Alexey

#ifndef MODULES_TASK_3_KRASILNIKOV_A_TRAPEZOID_METHOD_TRAPEZOID_METHOD_H_
#define MODULES_TASK_3_KRASILNIKOV_A_TRAPEZOID_METHOD_TRAPEZOID_METHOD_H_

#include <vector>

double trapezoidMethodSequential(double (*f)(double, double), double a_x, double b_x, double a_y, double b_y);
double trapezoidMethodParallel(double (*f)(double, double), double a_x, double b_x, double a_y, double b_y);

#endif  //  MODULES_TASK_3_KRASILNIKOV_A_TRAPEZOID_METHOD_TRAPEZOID_METHOD_H_
