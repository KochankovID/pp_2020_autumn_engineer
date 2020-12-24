// Copyright 2020 Krasilnikov Alexey

#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "../../../modules/task_3/krasilnikov_a_trapezoid_method/trapezoid_method.h"

void CustomTest(double (*f)(double, double), double a_x, double b_x, double a_y, double b_y);

double function_1(double x, double y) {
  return x * sin(y);
}

double function_2(double x, double y) {
  return x * y;
}

double function_3(double x, double y) {
  return pow(x, y);
}

double function_4(double x, double y) {
  return 0;
}

double function_5(double x, double y) {
  return 1;
}

TEST(Parallel_Operations_MPI, Test_function_1) {
  CustomTest(function_1, 0, 3, 0, 3);
}

TEST(Parallel_Operations_MPI, Test_function_2) {
  CustomTest(function_2, 0, 3, 0, 3);
}

TEST(Parallel_Operations_MPI, Test_function_3) {
  CustomTest(function_3, 0, 3, 0, 3);
}

TEST(Parallel_Operations_MPI, Test_function_4) {
  CustomTest(function_4, 0, 3, 0, 3);
}

TEST(Parallel_Operations_MPI, Test_function_5) {
  CustomTest(function_5, 0, 3, 0, 3);
}

void CustomTest(double (*f)(double, double), double a_x, double b_x, double a_y, double b_y) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double start, finish, parallel_time, sequential_time, parallel_res, sequential_res;
  if (rank == 0) {
    start = MPI_Wtime();
  }
  MPI_Barrier(MPI_COMM_WORLD);
  parallel_res = trapezoidMethodParallel(f, a_x, b_x, a_y, b_y);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    finish = MPI_Wtime();
    parallel_time = finish - start;
    start = MPI_Wtime();
    sequential_res = trapezoidMethodSequential(f, a_x, b_x, a_y, b_y);
    finish = MPI_Wtime();
    sequential_time = finish - start;
    std::cout << "------------------------------------------------------" << std::endl;
    std::cout << "--- Sequential time = " << sequential_time << std::endl;
    std::cout << "--- Parallel time = " << parallel_time << std::endl;
    std::cout << "--- Acceleration = " << sequential_time / parallel_time << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;
    ASSERT_EQ(std::abs(parallel_res - sequential_res) < 0.00001, true);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners =::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);

  return RUN_ALL_TESTS();
}
