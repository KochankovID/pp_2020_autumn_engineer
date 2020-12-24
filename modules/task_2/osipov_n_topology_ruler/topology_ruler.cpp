// Copyright 2020 Osipov Nikolay
#include "../../../modules/task_2/osipov_n_topology_ruler/topology_ruler.h"
#include <mpi.h>
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>


std::vector<int> getVector(int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<int> vector(size);
    for (int i = 0; i < size; i++) { vector[i] = gen() % 30; }
    return vector;
}

int getSequentialOperations(std::vector<int> vector, std::string operation) {
    const int  size = vector.size();
    int reduction_element = 0;
    if (operation == "+") {
        for (int i = 0; i < size; i++) {
            reduction_element += vector[i];
        }
    } else if (operation == "-") {
        for (int i = 0; i < size; i++) {
            reduction_element -= vector[i];
        }
    } else if (operation == "min") {
        reduction_element = vector[0];
        for (int  i = 1; i < size; i++) {
            reduction_element = std::min(reduction_element, vector[i]);
        }
    } else if (operation == "max") {
        reduction_element = vector[0];
        for (int i = 1; i < size; i++) {
            reduction_element = std::max(reduction_element, vector[i]);
        }
    }
    return reduction_element;
}

int getParallelOperations(std::vector<int> global_vector,
    int count_size_vector, std::string operation) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int delta = count_size_vector / size;

    if (rank == 0) {
        for (int proc = 1; proc < size; proc++) {
            MPI_Send(&global_vector[0] + proc * delta, delta,
                MPI_INT, proc, 0, MPI_COMM_WORLD);
        }
    }
    std::vector<int> local_vector(delta);
    if (rank == 0) {
        local_vector = std::vector<int>(global_vector.begin(),
            global_vector.begin() + delta);
    } else {
        MPI_Status status;
        MPI_Recv(&local_vector[0], delta, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    int global_sum = 0;
    int local_sum = getSequentialOperations(local_vector, operation);
    MPI_Op op;
    if (operation == "-" || operation == "+") { op = MPI_SUM; }
    if (operation == "min") { op = MPI_MIN; }
    if (operation == "max") { op = MPI_MAX; }
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, op, 0, MPI_COMM_WORLD);
    return global_sum;
}
