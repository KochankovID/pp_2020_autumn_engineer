// Copyright 2020 Gorbunova Valeria
#ifndef MODULES_TASK_3_GORBUNOVA_V_MONTE_CARLO_MONTE_CARLO_H_
#define MODULES_TASK_3_GORBUNOVA_V_MONTE_CARLO_MONTE_CARLO_H_
#include <string>

double MonteCarloSequential(double(*f)(double*), double a[],
    double b[], int n, int z);
double MonteCarloParallel(double(*f)(double*), double a[],
    double b[], int n, int z);
#endif  // MODULES_TASK_3_GORBUNOVA_V_MONTE_CARLO_MONTE_CARLO_H_
