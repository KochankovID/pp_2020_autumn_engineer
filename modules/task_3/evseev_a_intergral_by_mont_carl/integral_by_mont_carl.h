// Copyright 2020 Evseev Alexander
#ifndef MODULES_TASK_3_EVSEEV_A_INTERGRAL_BY_MONT_CARL_INTEGRAL_BY_MONT_CARL_H_
#define MODULES_TASK_3_EVSEEV_A_INTERGRAL_BY_MONT_CARL_INTEGRAL_BY_MONT_CARL_H_

#include <vector>

double MonteCarloParallel(std::vector<double> a, double sd,
                                   double (*pfunction)(std::vector<double>),
                                   bool (*parea)(std::vector<double>),
                                   unsigned int dm, int p_count);
double MonteCarloSequential(
    std::vector<double> a, double sd,
    double (*pfunction)(std::vector<double>), bool (*parea)(std::vector<double>),
    unsigned int dm, int p_count);

#endif  // MODULES_TASK_3_EVSEEV_A_INTERGRAL_BY_MONT_CARL_INTEGRAL_BY_MONT_CARL_H_
