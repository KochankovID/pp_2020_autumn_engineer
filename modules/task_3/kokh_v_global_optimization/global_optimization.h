// Copyright 2020 Kokh Vladislav
#ifndef MODULES_TASK_3_KOKH_V_GLOBAL_OPTIMIZATION_GLOBAL_OPTIMIZATION_H_
#define MODULES_TASK_3_KOKH_V_GLOBAL_OPTIMIZATION_GLOBAL_OPTIMIZATION_H_

#include <cmath>

double func1(double x, double y);

double func2(double x, double y);

double func3(double x, double y);

double func4(double x, double y);

double func5(double x, double y);


class setR {
 public:
    const double R;
    const double x;
    const double z;
    const double xPrev;
    const double zPrev;

    setR(const double r_value, const double x_value, const double z_value, const double x_valuePrev,
    const double _zPrev) :  R(r_value), x(x_value), z(z_value), xPrev(x_valuePrev), zPrev(_zPrev) {}

    friend bool operator<(const setR& left, const setR& right) {return left.R > right.R;}
};

struct xyzStruct {
    double x;
    double y;
    double z;
};

class setOne {
 public:
    const double x;
    const double y;

    setOne(double x_value, double y_value) : x(x_value), y(y_value) {}

    friend bool operator<(const setOne& left, const setOne& right) {return left.x < right.x;}
};

class setTwo {
 public:
    const double x;
    const double y;
    const double z;

    setTwo(const double x_value, const double y_value, const double z_value = 0) : x(x_value),
    y(y_value), z(z_value) {}

    friend bool operator<(const setTwo& left, const setTwo& right) {return left.x < right.x;}
};

bool comparingResults(const xyzStruct& a_value, const xyzStruct& b_value, const double& eps_value = 0.01);

xyzStruct forOne(const double& _a, const double& _b, const double& x_val_send,
double(*func)(double x, double y),
const double& eps_in_func = 0.1, const int& _N_max = 100, const double& _r_par = 2.0);

xyzStruct twoPar(const double& a1_value, const double& b1_value, const double& a2_value,
const double& b2_value,
double(*func)(double x, double y), const double& eps_in_func = 0.1, const int& _Nmax = 100,
const double& eps_one_value = 0.1,
const int& n_max_value_1 = 100, const double& _r_par = 2.0);

xyzStruct twoParSequential(const double& a1_value, const double& b1_value,
const double& a2_value, const double& b2_value,
double(*func)(double x, double y), const double& eps_in_func = 0.1, const int& _Nmax = 100,
const double& eps_one_value = 0.1,
const int& n_max_value_1 = 100, const double& _r_par = 2.0);

#endif  // MODULES_TASK_3_KOKH_V_GLOBAL_OPTIMIZATION_GLOBAL_OPTIMIZATION_H_
