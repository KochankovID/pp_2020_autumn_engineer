// Copyright 2020 Rachin Igor
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include "./integ_by_rect.h"

TEST(Parallel_Operations_MPI, Test_manual_integ) {
    int rank;
    double res1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        res1 = getSequentialInteg(f1, -1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.1, 0.1, 0.1);
        ASSERT_NEAR(res1, 4.98168, 0.01);  // due to step in 0.1
    }
}

TEST(Parallel_Operations_MPI, Test_parallel_integ) {
    int rank;
    double res1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    res1 = getParallelInteg(f1, -1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.1, 0.1, 0.1);
    if (rank == 0) {
        ASSERT_NEAR(res1, 4.98168, 0.01);  // due to step in 0.1
    }
}

TEST(Parallel_Operations_MPI, Test_parallel_with_manual) {
    int rank;
    double res1, res2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    res1 = getParallelInteg(f1, -1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.1, 0.1, 0.1);
    if (rank == 0) {
        res2 = getSequentialInteg(f1, -1.0, 0.0, 0.0, 2.0, 0.0, 1.0, 0.1, 0.1, 0.1);
        ASSERT_NEAR(res1, res2, 0.001);  // due to step in 0.1
    }
}

TEST(Parallel_Operations_MPI, Test_second_func) {
    int rank;
    double res1, res2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    res1 = getParallelInteg(f2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1, 0.1, 0.1);
    if (rank == 0) {
        res2 = getSequentialInteg(f2, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1, 0.1, 0.1);
        ASSERT_NEAR(res1, res2, 0.001);  // due to step in 0.1
    }
}

TEST(Parallel_Operations_MPI, Test_third_func) {
    int rank;
    double res1, res2;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    res1 = getParallelInteg(f3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1, 0.1, 0.1);
    if (rank == 0) {
        res2 = getSequentialInteg(f3, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.1, 0.1, 0.1);
        ASSERT_NEAR(res1, res2, 0.001);  // due to step in 0.1
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
