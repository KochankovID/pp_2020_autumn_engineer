// Copyright 2020 Shulman Egor
#include <mpi.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include "../../../modules/task_3/shulman_e_global_search_for_Strongin/global_search_for_Strongin.h"

double SequentialPart(const double x1, const double x2, double (*fcnPtr)(double),
                                    double epsilon, double r) {
    if (x1 == x2)
        throw "The intervals should be different";

    // Step 1
    std::vector<double> X;
    X.push_back(x1);
    X.push_back(x2);

    while (1) {
        // Step 2
        std::sort(X.begin(), X.end());

        // Step 3
        double maxlk;
        double Lk;
        double templk;
        for (int i = 0; i < static_cast<int>(X.size()) - 1; ++i) {
            templk = (std::abs(fcnPtr(X.at(i + 1)) - fcnPtr(X.at(i)))) / (X.at(i + 1) - X.at(i));
            if (i == 0)
                maxlk = templk;
            else
                maxlk = maxlk < templk ? templk : maxlk;
        }
        Lk = maxlk == 0 ? 1.0 : r * maxlk;

        // Step 4
        int Rt = 0;
        double minRi;
        double tempRi;
        for (int i = 0; i < static_cast<int>(X.size()) - 1; ++i) {
            tempRi = ((fcnPtr(X.at(i)) + fcnPtr(X.at(i + 1))) / 2) -
                            ((1 + pow((((fcnPtr(X.at(i + 1)) - fcnPtr(X.at(i))) / (X.at(i + 1) -
                                    X.at(i))) / Lk), 2)) / 2) * Lk * ((X.at(i + 1) - X.at(i)) / 2);
            if (i == 0) {
                minRi = tempRi;
            } else {
                if (minRi > tempRi) {
                    minRi = tempRi;
                    Rt = i;
                }
            }
        }

        // Step 5
        if (X.at(Rt + 1) - X.at(Rt) <= epsilon) {
            return X.at(Rt + 1);
        }

        // Step 6
        X.push_back((X.at(Rt) + X.at(Rt + 1)) / 2 - (fcnPtr(X.at(Rt + 1)) - fcnPtr(X.at(Rt))) / (2 * Lk));
    }
}

double root(const double x1, const double x2, double (*fcnPtr)(double), double epsilon, double r, int size) {
    MPI_Status status;

    // Step 1
    std::vector<double> X;
    X.push_back(x1);
    X.push_back(x2);

    while (1) {
        // Step 2
        std::sort(X.begin(), X.end());

        int block = static_cast<int>(X.size() - 1) / size;
        int remainder = static_cast<int>(X.size() - 1) % size;

        for (int i = 1; i < size; ++i)
            MPI_Send(&X[0] + remainder + i * block, block + 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

        // Step 3
        double maxlk;
        double Lk;
        double templk;
        for (int i = 0; i < block + remainder; ++i) {
            templk = (std::abs(fcnPtr(X.at(i + 1)) - fcnPtr(X.at(i)))) / (X.at(i + 1) - X.at(i));
            if (i == 0)
                maxlk = templk;
            else
                maxlk = maxlk < templk ? templk : maxlk;
        }
        Lk = maxlk <= 0 ? 1.0 : r * maxlk;

        if (block > 0) {
            for (int i = 1; i < size; i++) {
                MPI_Recv(&templk, 1, MPI_DOUBLE, i, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                Lk = Lk < templk ? templk : Lk;
            }
        }

        // Step 4
        int Rt = 0;
        double minRi;
        double tempRi;
        for (int i = 0; i < static_cast<int>(X.size()) - 1; ++i) {
            tempRi = ((fcnPtr(X.at(i)) + fcnPtr(X.at(i + 1))) / 2) -
                            ((1 + pow((((fcnPtr(X.at(i + 1)) - fcnPtr(X.at(i))) /
                                    (X.at(i + 1) - X.at(i))) / Lk), 2)) / 2) *
                                            Lk * ((X.at(i + 1) - X.at(i)) / 2);
            if (i == 0) {
                minRi = tempRi;
            } else {
                if (minRi > tempRi) {
                    minRi = tempRi;
                    Rt = i;
                }
            }
        }

        // Step 5
        if (X.at(Rt + 1) - X.at(Rt) <= epsilon) {
            for (int i = 1; i < size; ++i)
                MPI_Send(&X[0], block + 1, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
            return X.at(Rt + 1);
        }

        // Step 6
        double newX = (X.at(Rt) + X.at(Rt + 1)) / 2 - (fcnPtr(X.at(Rt + 1)) - fcnPtr(X.at(Rt))) / (2 * Lk);
        if (newX < X.at(0)) {
            Rt = 0;
            newX = (X.at(Rt) + X.at(Rt + 1)) / 2 - (fcnPtr(X.at(Rt + 1)) - fcnPtr(X.at(Rt))) / (2 * Lk);
        }
        if (newX > X.at(static_cast<int>(X.size()) - 1)) {
            Rt = static_cast<int>(X.size()) - 2;
            newX = (X.at(Rt) + X.at(Rt + 1)) / 2 - (fcnPtr(X.at(Rt + 1)) - fcnPtr(X.at(Rt))) / (2 * Lk);
        }
        X.push_back(newX);
    }
}

void other(const double x1, const double x2, double (*fcnPtr)(double), double epsilon, double r) {
    int block;
    MPI_Status status;
    while (1) {
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        MPI_Get_count(&status, MPI_DOUBLE, &block);

        std::vector<double> X(block + 1);
        MPI_Recv(&X[0], block + 1, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        if (status.MPI_TAG == 1)
            return;

        double localmaxlk, localLk = 1.0, localtemplk;
        if (block != 0) {
            for (int i = 0; i < static_cast<int>(X.size()) - 1; ++i) {
                localtemplk = (std::abs(fcnPtr(X.at(i + 1)) - fcnPtr(X.at(i)))) / (X.at(i + 1) - X.at(i));
                if (i == 0)
                    localmaxlk = localtemplk;
                else
                    localmaxlk = localmaxlk < localtemplk ? localtemplk : localmaxlk;
            }
            localLk = localmaxlk == 0 ? 1.0 : r * localmaxlk;
        }
        MPI_Send(&localLk, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
}


double ParallelPart(const double x1, const double x2, double (*fcnPtr)(double),
    double epsilon, double r) {
    if (x1 == x2)
        throw "The intervals should be different";

    double result;
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        result = root(x1, x2, fcnPtr, epsilon, r, size);
        return result;
    } else {
        other(x1, x2, fcnPtr, epsilon, r);
        return 0.0;
    }
}
