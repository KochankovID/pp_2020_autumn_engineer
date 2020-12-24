// Copyright 2020 Gorbunova Valeria
#include <mpi.h>
#include <string>
#include <algorithm>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <random>
#include <iostream>
#include <cstring>
#include "./../../modules/task_3/gorbunova_v_Monte_Carlo/monte_carlo.h"

double random(double min, double max) {
    return static_cast<double>(std::rand()) / RAND_MAX * (max - min) + min;
}

double MonteCarloSequential(double(*f)(double*), double a[],
    double b[], int n, int z) {
    double S = 1.0;
    double * s1 = new double[z];
    double S_integral = 0;
    for (int i = 0; i < z; i++) {
        S = S * (b[i] - a[i]);
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < z; j++) {
            s1[j] = random(a[j], b[j]);
        }
        S_integral += f(s1);
    }
    S = (S * S_integral)/n;
    return S;
}

double MonteCarloParallel(double(*f)(double*), double a[],
    double b[], int n, int z) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    double S = 1.0;
    double* s1 = new double[z];
    double S_integral = 0;
    double Local_S_integral = 0;
    for (int i = 0; i < n; i += size) {
        for (int j = 0; j < z; j++) {
            s1[j] = random(a[j], b[j]);
        }
        Local_S_integral += f(s1);
    }
    MPI_Reduce(&Local_S_integral, &S_integral, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        for (int i = 0; i < z; i++) {
            S = S * (b[i] - a[i]);
        }
        S = (S * S_integral) / n;
    }
    return S;
}
