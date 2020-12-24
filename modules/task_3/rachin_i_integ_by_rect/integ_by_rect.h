// Copyright 2020 Rachin Igor
#ifndef MODULES_TASK_3_RACHIN_I_INTEG_BY_RECT_INTEG_BY_RECT_H_
#define MODULES_TASK_3_RACHIN_I_INTEG_BY_RECT_INTEG_BY_RECT_H_

#include <string>

double f1(double x, double y, double z);
double f2(double x, double y, double z);
double f3(double x, double y, double z);
double getSequentialInteg(double f(double x, double y, double z),
    double x1, double x2, double y1, double y2, double z1, double z2, double hx, double hy, double hz);
double getParallelInteg(double f(double x, double y, double z),
    double x1, double x2, double y1, double y2, double z1, double z2, double hx, double hy, double hz);

#endif  // MODULES_TASK_3_RACHIN_I_INTEG_BY_RECT_INTEG_BY_RECT_H_
