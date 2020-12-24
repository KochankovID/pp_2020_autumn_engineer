// Copyright 2020 Evseev Alexander
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./integral_by_mont_carl.h"

double function_const(std::vector<double> args) { return 1; }

double func_quadratic(std::vector<double> args) { return args[0] * args[0]; }

double func_linear(std::vector<double> args) { return args[0] + args[1]; }

bool area_2_parabols(std::vector<double> args) {
  return (args[1] > (args[0] * args[0] - 1)) &&
         (args[1] < -args[0] * args[0] + 1);
}

bool intcircle(std::vector<double> args) {
  return (args[0] * args[0] + args[1] * args[1]) < 4;
}

bool intsphere1(std::vector<double> args) {
  return (args[0] * args[0] + args[1] * args[1] + args[2] * args[2]) < 4;
}

bool intsphere2(std::vector<double> args) {
  return (args[0] * args[0] + args[1] * args[1] + args[2] * args[2] +
          args[3] * args[3]) < 4;
}

bool intsphere3(std::vector<double> args) {
  return (args[0] > 0 && args[2] > 0 &&
          (args[0] * args[0] + args[1] * args[1] + args[2] * args[2]) < 4);
}

TEST(Monte_Karlo_MPI, TEST_const_func_int_three_sphere) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const double threr = 10.0;

  std::vector<double> s_point = {-2, -2, -2, -2};

  double integration_result = MonteCarloParallel(
      s_point, 4, function_const, intsphere2, 4, 10000);
  if (rank == 0) {
    ASSERT_NEAR(3.14 * 3.14 * 16 / 2, integration_result, threr);
  }
}

TEST(Monte_Karlo_MPI, TEST_quadr_func_int_sphere) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const double threr = 5.0;

  std::vector<double> s_point = {-2, -2, -2};

  double integration_result = MonteCarloParallel(
      s_point, 4, func_quadratic, intsphere3, 3, 10000);
  if (rank == 0) {
    ASSERT_NEAR(3.14 * 32 / 15, integration_result, threr);
  }
}

TEST(Monte_Karlo_MPI, TEST_const_func_intsphere) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const double threr = 5.0;

  std::vector<double> s_point = {-2, -2, -2};

  double integration_result = MonteCarloParallel(
      s_point, 4, function_const, intsphere1, 3, 10000);
  if (rank == 0) {
    ASSERT_NEAR(3.14 * 4 * 8 / 3, integration_result, threr);
  }
}

TEST(Monte_Karlo_MPI, TEST_linear_func_int_2_parabols) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const double threr = 1.0;

  std::vector<double> s_point = {-1, -1};

  double integration_result = MonteCarloParallel(
      s_point, 2, func_linear, area_2_parabols, 2, 10000);

  if (rank == 0) {
    ASSERT_NEAR(0, integration_result, threr);
  }
}

TEST(Monte_Karlo_MPI, TEST_const_func_intcircle) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const double threr = 1.0;

  std::vector<double> s_point = {-2, -2};

  double integration_result = MonteCarloParallel(
      s_point, 4, function_const, intcircle, 2, 10000);
  if (rank == 0) {
    ASSERT_NEAR(3.14 * 4, integration_result, threr);
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
