// Copyright 2020 Rachin Igor
#define _USE_MATH_DEFINES

#include <mpi.h>
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <iostream>
#include "../../modules/task_3/rachin_i_integ_by_rect/integ_by_rect.h"

double f1(double x, double y, double z) {
    return(8*pow(y, 2)*z*pow(M_E, 2*x*y*z));
}

double f2(double x, double y, double z) {
    return(pow(x, 8)*pow(y, 7)*pow(z, 4));
}

double f3(double x, double y, double z) {
    return(pow(x, 8) * pow(y, 7) / pow(z, 5));
}

double getSequentialInteg(double f(double x, double y, double z),
    double x1, double x2, double y1, double y2, double z1, double z2, double hx, double hy, double hz) {
    double result = 0.0;
    double x, y, z;
    int px = static_cast<int>((x2 - x1) / hx);
    int py = static_cast<int>((y2 - y1) / hy);
    int pz = static_cast<int>((z2 - z1) / hz);
    for (int i = 0; i < px; i++) {
        for (int j = 0; j < py; j++) {
            for (int k = 0; k < pz; k++) {
                x = x1 + i * hx + hx / 2;
                y = y1 + j * hy + hy / 2;
                z = z1 + k * hz + hz / 2;
                result += f(x, y, z);
            }
        }
    }
    return hx*hy*hz*result;
}

double getParallelInteg(double f(double x, double y, double z),
    double x1, double x2, double y1, double y2, double z1, double z2, double hx, double hy, double hz) {
    int size, rank;
    double result = 0.0;
    double localRes = 0.0;
    double x, y, z;
    int px = static_cast<int>((x2 - x1) / hx);
    int py = static_cast<int>((y2 - y1) / hy);
    int pz = static_cast<int>((z2 - z1) / hz);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int delta = pz / size;
    for (int i = 0; i < px; i++) {
        for (int j = 0; j < py; j++) {
            if (rank != size - 1) {
                for (int k = delta*rank; k < delta*(rank+1); k++) {
                    x = x1 + i * hx + hx / 2;
                    y = y1 + j * hy + hy / 2;
                    z = z1 + k * hz + hz / 2;
                    localRes += f(x, y, z);
                }
            } else {
                for (int k = delta*rank; k < pz; k++) {
                    x = x1 + i * hx + hx / 2;
                    y = y1 + j * hy + hy / 2;
                    z = z1 + k * hz + hz / 2;
                    localRes += f(x, y, z);
                }
            }
        }
    }
    MPI_Reduce(&localRes, &result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return hx*hy*hz*result;
}
