// Copyright 2020 Kochankov Ilya
#include <mpi.h>
#include <algorithm>
#include <vector>
#include <random>
#include <ctime>
#include "../../../modules/task_1/kochankov_i_matrix_columns_sum/matrix_columns_sum.h"


double* sequential_operations(int result_size, int start_index, const vector<vector<double>>& matrix) {
    auto* tmp_result_vector = new double[result_size];
    std::fill(tmp_result_vector, tmp_result_vector+result_size, 0);

    for (int j = start_index; j < start_index + result_size; j++) {
        for (auto &i : matrix) {
            tmp_result_vector[j-start_index] += i[j];
        }
    }

    return tmp_result_vector;
}

vector<double> parallel_sum(const vector<vector<double >>& matrix) {
    int size, rank, len = matrix[0].size();

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    auto *result_vector = new double[len];

    const int delta = len / size;
    int start_index = delta * rank;

    int result_size;

    if (delta * (rank + 1) > len) {
        result_size = len - start_index;
    } else {
        result_size = delta;
    }

    auto tmp_result_vector = sequential_operations(result_size, start_index, matrix);

    MPI_Gather(tmp_result_vector, result_size, MPI_DOUBLE,
            result_vector+(rank*delta), result_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    delete [] tmp_result_vector;

    vector<double> columns_sums(result_vector, result_vector+len);
    delete [] result_vector;

    return columns_sums;
}

vector<vector<double >> get_rand_matrix(int x, int y) {
    std::mt19937 gen;

    gen.seed(static_cast<unsigned int>(time(0)));
    vector<vector<double >> vec(x);
    for (int i = 0; i < x; i++) {
        vec[i].resize(y);
        for (int j = 0; j < y; j++) {
            vec[i][j] = (static_cast<double>(gen()) / 100000);
        }
    }
    return vec;
}

vector<double> not_parallel_sum(const vector<vector<double>> &matrix) {
    vector<double> result_vector(matrix[0].size());

    for (const auto & i : matrix) {
        for (int j = 0; j < static_cast<int>(i.size()); j++) {
            result_vector[j] += i[j];
        }
    }
    return result_vector;
}
