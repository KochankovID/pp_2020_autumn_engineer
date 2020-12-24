// Copyright 2020 Kokh Vladislav
#define _USE_MATH_DEFINES

#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include "./global_optimization.h"


TEST(Global_Optimization_MPI, Test_1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double(*ptr)(double, double) = func1;

    xyzStruct result = twoPar(-4, 4, -4, 4, ptr);
    if (rank == 0) {
        xyzStruct just_result = twoParSequential(-4, 4, -4, 4, ptr);
        ASSERT_TRUE(comparingResults(result, just_result, 0.1));
    }
}

TEST(Global_Optimization_MPI, Test_2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double(*ptr)(double, double) = func2;

    xyzStruct result = twoPar(-0.9, 1, -1, 1, ptr);

    if (rank == 0) {
        xyzStruct must_be_result = {0, 0, 4};
        ASSERT_TRUE(comparingResults(result, must_be_result, 0.1));
    }
}

TEST(Global_Optimization_MPI, Test_3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double(*ptr)(double, double) = func3;

    xyzStruct result = twoPar(0, 2, 0.3, 1, ptr);
    if (rank == 0) {
        xyzStruct must_be_result = {1, 0.5, 4};
        ASSERT_TRUE(comparingResults(result, must_be_result, 0.1));
    }
}

TEST(Global_Optimization_MPI, Test_4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double(*ptr)(double, double) = func4;

    xyzStruct result = twoPar(0.1, 5, -5, 5, ptr);
    if (rank == 0) {
        xyzStruct must_be_result = {5, -5, -136.18};
        ASSERT_TRUE(comparingResults(result, must_be_result, 0.1));
    }
}

TEST(Global_Optimization_MPI, Test_5) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double(*ptr)(double, double) = func5;

    xyzStruct result = twoPar(1, 3, 1, 2, ptr);
    if (rank == 0) {
        xyzStruct must_be_result = {2, 1.25, -1.5};
        ASSERT_TRUE(comparingResults(result, must_be_result, 0.1));
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
