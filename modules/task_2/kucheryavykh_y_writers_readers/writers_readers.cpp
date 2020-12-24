// Copyright 2020 Kucheryavykh Yan
#include <mpi.h>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <iostream>
#include "../../../modules/task_2/kucheryavykh_y_writers_readers/writers_readers.h"

#define READ 2
#define WRITE 4

void wait_random_seconds() {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    double t1 = MPI_Wtime();
    while (MPI_Wtime() - t1 < 0.05 + static_cast<float>(gen() % 100) / 10) {}
}

int main_process(int loops) {
    MPI_Request request;
    MPI_Status status;

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int get;
    int variable = 0;
    while (loops != 0) {
        MPI_Recv(&get, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        --loops;
        if (status.MPI_TAG == WRITE) {
            ++variable;
            std::cout << "WRITE: " << variable << " process: " << status.MPI_SOURCE << std::endl;
            MPI_Isend(&variable, 1, MPI_INT, status.MPI_SOURCE, WRITE, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        } else {
            std::cout << "READ: " << variable << " process: " << status.MPI_SOURCE << std::endl;
            MPI_Isend(&variable, 1, MPI_INT, status.MPI_SOURCE, READ, MPI_COMM_WORLD, &request);
            MPI_Request_free(&request);
        }
    }
    return variable;
}

void writer(int times) {
    int rank, variable;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < times; i++) {
        wait_random_seconds();
        MPI_Send(&variable, 1, MPI_INT, 0, WRITE, MPI_COMM_WORLD);
        MPI_Recv(&variable, 1, MPI_INT, 0, WRITE, MPI_COMM_WORLD, &status);
    }
}

void reader(int times) {
    int rank, variable;
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    for (int i = 0; i < times; i++) {
        wait_random_seconds();
        MPI_Send(&variable, 1, MPI_INT, 0, READ, MPI_COMM_WORLD);
        MPI_Recv(&variable, 1, MPI_INT, 0, READ, MPI_COMM_WORLD, &status);
    }
}
