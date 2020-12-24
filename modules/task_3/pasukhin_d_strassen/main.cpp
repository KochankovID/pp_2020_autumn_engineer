// Copyright 2020 Pasukhin Dmitry

#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <limits>
#include "./strassen.h"

const bool debug = false;

TEST(Parallel_Operations_MPI, Test_Little_Strassen_Result) {
  int comm_rank;
  int comm_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int pow_size = 4*comm_size;
  const int size = pow_size*pow_size;
  double* A = new double[size];
  GenerateMatrix(A, size, 2);

  double* B = new double[size];
  GenerateMatrix(B, size, 3);


  double* res_C = new double[size];
  double* org_C = new double[size];

  const double start = MPI_Wtime();
  Strassen_Multiplication(A, B, res_C, pow_size);
  if (debug && comm_rank == 0)
    std::cout << "Paral time: " << MPI_Wtime() - start << std::endl;

  const double start2 = MPI_Wtime();
  Serial_Strassen(A, B, org_C, pow_size);
  if (debug && comm_rank == 0)
    std::cout << "Seq time: " << MPI_Wtime() - start2 << std::endl;

  if (comm_rank == 0) {
    bool is_OK = true;

    for (int i = 0; i < size; ++i) {
      if (fabs(org_C[i] - res_C[i]) >= std::numeric_limits<double>::epsilon())
        is_OK = false;
    }

    ASSERT_EQ(true, is_OK);
  }
}

TEST(Parallel_Operations_MPI, Test_Middle_Strassen_Result) {
  int comm_rank;
  int comm_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int pow_size = 16 * comm_size;
  const int size = pow_size * pow_size;

  double* A = new double[size];
  GenerateMatrix(A, size, 2);

  double* B = new double[size];
  GenerateMatrix(B, size, 3);

  double* res_C = new double[size];
  double* org_C = new double[size];

  const double start = MPI_Wtime();
  Strassen_Multiplication(A, B, res_C, pow_size);
  if (debug && comm_rank == 0)
    std::cout << "Paral time: " << MPI_Wtime() - start << std::endl;

  const double start2 = MPI_Wtime();
  Serial_Strassen(A, B, org_C, pow_size);
  if (debug && comm_rank == 0)
    std::cout << "Seq time: " << MPI_Wtime() - start2 << std::endl;

  if (comm_rank == 0) {
    bool is_OK = true;

    for (int i = 0; i < size; ++i) {
      if (fabs(org_C[i] - res_C[i]) >= std::numeric_limits<double>::epsilon())
        is_OK = false;
    }

    ASSERT_EQ(true, is_OK);
  }
}

TEST(Parallel_Operations_MPI, Test_Big_Strassen_result) {
  int comm_rank;
  int comm_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int pow_size = 64 * comm_size/4;
  const int size = pow_size * pow_size;

  double* A = new double[size];
  GenerateMatrix(A, size, 2);

  double* B = new double[size];
  GenerateMatrix(B, size, 3);

  double* res_C = new double[size];
  double* org_C = new double[size];

  const double start = MPI_Wtime();
  Strassen_Multiplication(A, B, res_C, pow_size);
  if (debug && comm_rank == 0)
    std::cout << "Paral time: " << MPI_Wtime() - start << std::endl;

  const double start2 = MPI_Wtime();
  Serial_Strassen(A, B, org_C, pow_size);
  if (debug && comm_rank == 0)
    std::cout << "Seq time: " << MPI_Wtime() - start2 << std::endl;

  if (comm_rank == 0) {
    bool is_OK = true;

    for (int i = 0; i < size; ++i) {
      if (fabs(org_C[i] - res_C[i]) >= std::numeric_limits<double>::epsilon())
        is_OK = false;
    }

    ASSERT_EQ(true, is_OK);
  }
}

TEST(Parallel_Operations_MPI, Test_Static_Middle_Strassen_Compare) {
  int comm_rank;
  int comm_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  const std::vector<double> A = { 4.25, 1.75, 8.50, 1.,
                                  1.25, 6.75, 5.50, 2.,
                                  5.25, 4.75, 9.50, 3.,
                                  1.25, 6.75, 3.50, 4. };

  const std::vector<double> B = { 2.25, 1.75, 3.50, 4.,
                                  1.25, 8.75, 5.50, 3.,
                                  6.25, 2.75, 9.50, 2.,
                                  9.25, 1.75, 8.50, 1., };

  const std::vector<double> C = { 74.125,  47.875, 113.75, 40.25,
                                  64.125,  79.875, 110.75, 38.25,
                                  104.875, 82.125, 160.25, 57.25,
                                  70.125,  77.875, 108.75, 36.25 };

  double* res_C = new double[16];

  Strassen_Multiplication(&A.at(0), &B.at(0), res_C, 4);

  if (comm_rank == 0) {
    bool is_OK = true;

    for (int i = 0; i < 16; ++i) {
      if (fabs(C[i] - res_C[i]) >= std::numeric_limits<double>::epsilon())
        is_OK = false;
    }

    ASSERT_EQ(true, is_OK);
  }
}

TEST(Parallel_Operations_MPI, Test_Static_Little_Strassen_Compare) {
  int comm_rank;
  int comm_size;

  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  const std::vector<double> A = { 4.25, 1.75,
                                  1.25, 6.75, };

  const std::vector<double> B = { 2.25, 3.75,
                                  1.75, 8.75 };

  const std::vector<double> C = { 12.625, 31.25,
                                  14.625, 63.75 };

  double* res_C = new double[4];

  Strassen_Multiplication(&A.at(0), &B.at(0), res_C, 2);

  if (comm_rank == 0) {
    bool is_OK = true;

    for (int i = 0; i < 4; ++i) {
      if (fabs(C[i] - res_C[i]) >= std::numeric_limits<double>::epsilon())
        is_OK = false;
    }

    ASSERT_EQ(true, is_OK);
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
