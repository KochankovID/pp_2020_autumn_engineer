// Copyright 2020 Shulman Egor
#ifndef MODULES_TASK_3_SHULMAN_E_GLOBAL_SEARCH_FOR_STRONGIN_GLOBAL_SEARCH_FOR_STRONGIN_H_
#define MODULES_TASK_3_SHULMAN_E_GLOBAL_SEARCH_FOR_STRONGIN_GLOBAL_SEARCH_FOR_STRONGIN_H_

double SequentialPart(const double x1, const double x2, double (*fcnPtr)(double),
                                double epsilon, double r);

double ParallelPart(const double x1, const double x2, double (*fcnPtr)(double),
                                double epsilon, double r);

double root(const double x1, const double x2, double (*fcnPtr)(double),
                                double epsilon, double r, int size);

void other(const double x1, const double x2, double (*fcnPtr)(double),
                                double epsilon, double r);

#endif  // MODULES_TASK_3_SHULMAN_E_GLOBAL_SEARCH_FOR_STRONGIN_GLOBAL_SEARCH_FOR_STRONGIN_H_
