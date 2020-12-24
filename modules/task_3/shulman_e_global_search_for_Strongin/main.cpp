// Copyright 2020 Shulman Egor
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include "./global_search_for_Strongin.h"

inline double Function(double x) {
    return (10 + x*sin(x));
}

TEST(Parallel_Operations_MPI, SequentialFunc1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double (*fcnPtr)(double) = Function;
    if (rank == 0) {
        double Result = SequentialPart(1., 310.0, fcnPtr, 1e-3, 2);
        double Q = -296.307;
        ASSERT_NEAR(fcnPtr(Result), Q, 0.1);
    }
}

TEST(Parallel_Operations_MPI, ParallelFunc1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double (*fcnPtr)(double) = Function;
    double Result = ParallelPart(1., 310.0, fcnPtr, 1e-3, 2);
    if (rank == 0) {
        double Q = -296.307;
        ASSERT_NEAR(fcnPtr(Result), Q, 0.1);
    }
}

TEST(Parallel_Operations_MPI, ThrowEquality) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double (*fcnPtr)(double) = Function;
    if (rank == 0) {
        EXPECT_ANY_THROW(ParallelPart(1., 1., fcnPtr, 1e-3, 2));
    }
}

inline double Function2(double x) {
    return (x + 8 / pow(x, 4));
}

TEST(Parallel_Operations_MPI, Eps_3_Func2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double (*fcnPtr)(double) = Function2;
    double Result = ParallelPart(1., 100., fcnPtr, 1e-3, 2);

    if (rank == 0) {
        double Q = 2.5;
        ASSERT_NEAR(fcnPtr(Result), Q, 0.1);
    }
}

TEST(Parallel_Operations_MPI, Eps_5_Func2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double (*fcnPtr)(double) = Function2;
    double Result = ParallelPart(1., 100., fcnPtr, 1e-5, 2);

    if (rank == 0) {
        double Q = 2.5;
        ASSERT_NEAR(fcnPtr(Result), Q, 0.1);
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
