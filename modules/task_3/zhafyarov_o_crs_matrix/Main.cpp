  // Copyright 2020 Zhafyarov Oleg
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include "./crs_matrix.h"

TEST(Parallel_Operations_MPI, FirstTest_Matrix_CRS_Random_100x100) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 100;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  ParallelMultiplication(process_rank, Size, &A, &B, &ToCompare);
  bool IsCompare = true;
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Pointer[i] != ToCompare.Pointer[i]) {
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, SecondTest_Matrix_CRS_Random_150x150) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 150;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  bool IsCompare = true;
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Col[i] < 0) {
        std::cout << Result.Col[i];
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, ThirdTest_Matrix_CRS_Random_250x250) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 250;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  bool IsCompare = true;
  ParallelMultiplication(process_rank, Size, &A, &B, &ToCompare);
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Pointer[i] != ToCompare.Pointer[i]) {
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, FourthTest_Matrix_CRS_Random_500x500) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 500;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS A, B;
  MatrixCRS Result;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  bool IsCompare = true;
  if (process_rank == 0) {
    for (int i = 0; i < All_No_empty; i++) {
      if (Result.Col[i] < 0) {
        IsCompare = false;
        break;
      }
    }
    ASSERT_EQ(IsCompare, true);
  }
}

TEST(Parallel_Operations_MPI, FifthTest_efficiency_2000x2000) {
  int process_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);
  int Size = 2000;
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;
  double start_time, end_time, parallel, sequantional;
  MatrixCRS A, B;
  MatrixCRS Result;
  MatrixCRS ToCompare;
  InitializeMatrix(Size, All_No_empty, &Result);
  InitializeMatrix(Size, All_No_empty, &A);
  InitializeMatrix(Size, All_No_empty, &B);
  InitializeMatrix(Size, All_No_empty, &ToCompare);
  if (process_rank == 0) {
    RandomMatrixCRS(Size, Nonzero_in_row, &A);
    RandomMatrixCRS(Size, Nonzero_in_row, &B);
    B = Transpose(Size, All_No_empty, Nonzero_in_row, &B);
  }
  start_time = MPI_Wtime();
  ParallelMultiplication(process_rank, Size, &A, &B, &Result);
  end_time = MPI_Wtime();
  parallel = end_time - start_time;
  bool IsCompare = true;
  if (process_rank == 0) {
    start_time = MPI_Wtime();
    Multiplication(&A, &B, &ToCompare);
    end_time = MPI_Wtime();
    sequantional = end_time - start_time;
    std::cout << "Parallel = " << parallel << std::endl;
    std::cout << "Sequantional = " << sequantional << std::endl;
    ASSERT_EQ(IsCompare, true);
  }
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);

  AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());

  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}


