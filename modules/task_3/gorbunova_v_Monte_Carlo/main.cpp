// Copyright 2020 Gorbunova Valeria
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <cmath>
#include <string>
#include "./monte_carlo.h"

double integral(double x[]) {
    return x[0] * x[0] * x[0] - x[1] * x[1];
}

double integral1(double x[]) {
    return x[0] - x[1];
}

double integral2(double x[]) {
    return x[0] * x[0] + x[0] * x[1] + x[1] * x[1];
}

double integral3(double x[]) {
    return  sin(x[1] * x[2] * x[0]);
}

double integral4(double x[]) {
    return 5 * x[0] * x[2] / x[1] * x[3];
}

TEST(Monte_carlol_test, test_result_of_integral) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double a[2] = { 1.1, 2.34 };
    double b[2] = { 4.2 , 3.0 };
    if (rank == 0) {
        double res = MonteCarloSequential(integral, a, b, 100000, 2);
        ASSERT_NEAR(res, 36.439, 0.5);
    }
}

TEST(Monte_carlol_test, test_integral_1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double a[2] = { 1.3, 1.5 };
    double b[2] = { 3.2, 2.0 };
    double MPIMonteCarloStart = MPI_Wtime();
    double res1 = MonteCarloParallel(integral1, a, b, 100000, 2);
    double MPIMonteCarloEnd = MPI_Wtime();
    if (rank == 0) {
        double MonteCarloStart = MPI_Wtime();
        double res = MonteCarloSequential(integral1, a, b, 100000, 2);
        double MonteCarloEnd = MPI_Wtime();
        std::cout << std::fixed << std::setprecision(8) << "My_scatter :    "
            << MonteCarloEnd - MonteCarloStart << std::endl;
        std::cout << std::fixed << std::setprecision(8) << "MPIscatter :    "
            << MPIMonteCarloEnd - MPIMonteCarloStart << std::endl;
        ASSERT_NEAR(res1, res, 0.5);
  }
}


TEST(Monte_carlol_test, test_integral_2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double a[2] = { 0.0, 1.0 };
    double b[2] = { 0.4 , 2.5 };
    double MPIMonteCarloStart = MPI_Wtime();
    double res1 = MonteCarloParallel(integral2, a, b, 100000, 2);
    double MPIMonteCarloEnd = MPI_Wtime();
    if (rank == 0) {
        double MonteCarloStart = MPI_Wtime();
        double res = MonteCarloSequential(integral2, a, b, 100000, 2);
        double MonteCarloEnd = MPI_Wtime();
        std::cout << std::fixed << std::setprecision(8) << "My_scatter :    "
            << MonteCarloEnd - MonteCarloStart << std::endl;
        std::cout << std::fixed << std::setprecision(8) << "MPIscatter :    "
            << MPIMonteCarloEnd - MPIMonteCarloStart << std::endl;
        ASSERT_NEAR(res1, res, 0.5);
    }
}


TEST(Monte_carlol_test, test_integral_3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double a[3] = { 0.0 , 0.5 , 0.0};
    double b[3] = { 2.0 , 3.14 , 1.0};
    double MPIMonteCarloStart = MPI_Wtime();
    double res1 = MonteCarloParallel(integral3, a, b, 100000, 2);
    double MPIMonteCarloEnd = MPI_Wtime();
    if (rank == 0) {
        double MonteCarloStart = MPI_Wtime();
        double res = MonteCarloSequential(integral3, a, b, 100000, 2);
        double MonteCarloEnd = MPI_Wtime();
        std::cout << std::fixed << std::setprecision(8) << "My_scatter :    "
            << MonteCarloEnd - MonteCarloStart << std::endl;
        std::cout << std::fixed << std::setprecision(8) << "MPIscatter :    "
            << MPIMonteCarloEnd - MPIMonteCarloStart << std::endl;
        ASSERT_NEAR(res1, res, 0.5);
    }
}

TEST(Monte_carlol_test, test_integral_4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double a[4] = { 0.0 , 0.34 , 0.9 , 1.0 };
    double b[4] = { 1.3 , 2.2 , 3.0, 2.0 };
    double MPIMonteCarloStart = MPI_Wtime();
    double res1 = MonteCarloParallel(integral4, a, b, 100000, 2);
    double MPIMonteCarloEnd = MPI_Wtime();
    if (rank == 0) {
        double MonteCarloStart = MPI_Wtime();
        double res = MonteCarloSequential(integral4, a, b, 100000, 2);
        double MonteCarloEnd = MPI_Wtime();
        std::cout << std::fixed << std::setprecision(8) << "My_scatter :    "
            << MonteCarloEnd - MonteCarloStart << std::endl;
        std::cout << std::fixed << std::setprecision(8) << "MPIscatter :    "
            << MPIMonteCarloEnd - MPIMonteCarloStart << std::endl;
        ASSERT_NEAR(res1, res, 0.5);
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
