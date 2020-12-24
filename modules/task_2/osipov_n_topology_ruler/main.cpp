// Copyright 2020 Osipov Nikolay
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_2/osipov_n_topology_ruler/topology_ruler.h"

TEST(MPILinearTopology, Test_Sum) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vector;
    const int count_size_vector = 120;

    if (rank == 0) {
        global_vector = getVector(count_size_vector);
    }

    int parallel_sum = getParallelOperations(global_vector, count_size_vector, "+");

    if (rank == 0) {
        int sequential_sum = getSequentialOperations(global_vector, "+");
        ASSERT_EQ(sequential_sum, parallel_sum);
    }
}

TEST(MPILinearTopology, Test_Difference) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vector;
    const int count_size_vector = 120;

    if (rank == 0) {
        global_vector = getVector(count_size_vector);
    }

    int parallel_difference = getParallelOperations(global_vector, count_size_vector, "-");

    if (rank == 0) {
        int sequential_difference = getSequentialOperations(global_vector, "-");
        ASSERT_EQ(sequential_difference, parallel_difference);
    }
}

TEST(MPILinearTopology, Test_Max) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vector;
    const int count_size_vector = 120;

    if (rank == 0) {
        global_vector = getVector(count_size_vector);
    }

    int parallel_max;
    parallel_max = getParallelOperations(global_vector, count_size_vector, "max");

    if (rank == 0) {
        int sequential_max = getSequentialOperations(global_vector, "max");
        ASSERT_EQ(sequential_max, parallel_max);
    }
}

TEST(MPILinearTopology, Test_Min) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vector;
    const int count_size_vector = 120;

    if (rank == 0) {
        global_vector = getVector(count_size_vector);
    }

    int parallel_min;
    parallel_min = getParallelOperations(global_vector, count_size_vector, "min");

    if (rank == 0) {
        int sequential_min = getSequentialOperations(global_vector, "min");
        ASSERT_EQ(sequential_min, parallel_min);
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
