// Copyright 2018 Nesterov Alexander

#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./radix_sort_simple_merge.h"

TEST(Parallel_Operations_MPI, Test_Negative_Size) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec;
  const int size = -666;
  if (rank == 0) {
  ASSERT_ANY_THROW(getRandomVector(size));
  }
}

TEST(Parallel_Operations_MPI, Test_Get_Random_Array) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int size = 100;
  if (rank == 0) {
  ASSERT_NO_THROW(getRandomVector(size));
  }
}

TEST(Parallel_Operations_MPI, TEST_RadixSort_simple) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec{7, 9, 8, 5, 4, 7, 7};
  std::vector<int> res{4, 5, 7, 7, 7, 8, 9};
  if (rank == 0) {
     vec = radixSort(vec, 1);
     for (int i = 0; i < 7; ++i)
         EXPECT_EQ(vec[i], res[i]) << "Arrays check_array and res_array differ at index " << i;
  }
}

TEST(Parallel_Operations_MPI, TEST_Check_First_Radix) {
int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec{71, 92, 86, 54, 43, 72, 77};
  std::vector<int> res{71, 92, 72, 43, 54, 86, 77};
  if (rank == 0) {
     vec = radixSort(vec, 1);
     for (int i = 0; i < 7; ++i)
         EXPECT_EQ(vec[i], res[i]) << "Arrays check_array and res_array differ at index " << i;
  }
}

TEST(Parallel_Operations_MPI, TEST_Check_Two_Radix) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec{71, 92, 86, 54, 43, 72, 77};
  std::vector<int> res{43, 54, 71, 72, 77, 86, 92};
  if (rank == 0) {
     vec = radixSort(vec, 1);
     vec = radixSort(vec, 2);
     for (int i = 0; i < 7; ++i)
         EXPECT_EQ(vec[i], res[i]) << "Arrays check_array and res_array differ at index " << i;
  }
}

TEST(Parallel_Operations_MPI, TEST_Check_Three_Radix) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec{15, 192, 82, 387, 5, 634, 34};
  std::vector<int> res{5, 15, 34, 82, 192, 387, 634};
  if (rank == 0) {
     vec = radixSort(vec, 1);
     vec = radixSort(vec, 2);
     vec = radixSort(vec, 3);
     for (int i = 0; i < 7; ++i)
         EXPECT_EQ(vec[i], res[i]) << "Arrays check_array and res_array differ at index " << i;
  }
}

TEST(Parallel_Operations_MPI, TEST_Check_Merge) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
     std::vector<int> vec1{1, 1, 3, 5};
     std::vector<int> vec2{0, 2, 3, 6};
     std::vector<int> res{0, 1, 1, 2, 3, 3, 5, 6};
     std::vector<int> vec3 = merge(vec1, vec2);
     for (int i = 0; i < 8; ++i)
         EXPECT_EQ(vec3[i], res[i]) << "Arrays check_array and res_array differ at index " << i;
  }
}

TEST(Parallel_Operations_MPI, TEST_ParallelRadixSort) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec{7, 9, 8, 5, 4, 7, 7, 2, 5, 8, 0, 6, 7, 1, 9, 2, 5, 6, 7, 1};
  std::vector<int> res = ParallelRadixSort(vec, 20);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
     std::vector<int> check = checkVector(vec);
     for (int i = 0; i < 20; ++i)
         EXPECT_EQ(res[i], check[i]) << "Arrays check_array and res_array differ at index " << i;
  }
}

TEST(Parallel_Operations_MPI, TEST_ParallelRadixSort2) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec{7, 9, 8, 5, 4, 7, 7, 2, 5, 8, 0, 6, 7};
  std::vector<int> res = ParallelRadixSort(vec, 13);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
     std::vector<int> check = checkVector(vec);
     for (int i = 0; i < 13; ++i)
         EXPECT_EQ(res[i], check[i]) << "Arrays check_array and res_array differ at index " << i;
  }
}

TEST(Parallel_Operations_MPI, TEST_ParallelRadixSort3) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec{73, 91, 817, 5, 42, 74, 701, 23, 59, 8, 0, 66, 773, 1024};
  std::vector<int> res = ParallelRadixSort(vec, 14);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
     std::vector<int> check = checkVector(vec);
     for (int i = 0; i < 14; ++i)
         EXPECT_EQ(res[i], check[i]) << "Arrays check_array and res_array differ at index " << i;
  }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);
    // difOfTime(20000001);
    // timeOfRadixSort(10000001);
    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
