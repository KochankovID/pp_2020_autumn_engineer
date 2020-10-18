// Copyright 2020 Kochankov Ilya
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./matrix_columns_sum.h"


TEST(Parallel_Operations_MPI, parallel_sum_works) {
    vector<vector<double >> matrix = { {1, 5, 2, 3}, {1, 5, 2, 3}, {1, 5, 2, 3} };
    vector<double> result_vector = {3, 15, 6, 9};

    auto result = parallel_sum(matrix);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        EXPECT_EQ(result, result_vector);
    }
}

TEST(Parallel_Operations_MPI, not_parallel_sum_works) {
    vector<vector<double >> matrix = { {1, 5, 2, 3}, {1, 5, 2, 3}, {1, 5, 2, 3} };
    vector<double> result_vector = {3, 15, 6, 9};

    auto result = not_parallel_sum(matrix);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_EQ(result, result_vector);
    }
}

TEST(Parallel_Operations_MPI, parallel_sum_one_line) {
    vector<vector<double >> matrix = {{1, 5, 2, 3}};
    vector<double> result_vector = {1, 5, 2, 3};

    auto result = parallel_sum(matrix);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_EQ(result, result_vector);
    }
}

TEST(Parallel_Operations_MPI, get_rand_matrix_works) {
    auto matrix = get_rand_matrix(5, 6);

    EXPECT_EQ(matrix.size(), 5);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(matrix[i].size(), 6);
        for (int j = 0; j < 6; j++) {
            EXPECT_GE(matrix[i][j], 0);
        }
    }
}

TEST(Parallel_Operations_MPI, get_rand_matrix_negative_size) {
    EXPECT_ANY_THROW(auto matrix = get_rand_matrix(-5, -9));
}

TEST(Parallel_Operations_MPI, parallel_sum_works_random_matrix) {
    vector<vector<double >> matrix = get_rand_matrix(10000, 5000);

    auto result_vector = not_parallel_sum(matrix);

    auto parallel_result = parallel_sum(matrix);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        EXPECT_EQ(parallel_result, result_vector);
    }
}


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners &listeners =
            ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
