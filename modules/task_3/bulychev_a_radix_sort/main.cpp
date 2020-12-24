// Copyright 2020 Bulychev Andrey
#include <gtest/gtest.h>

#include <gtest-mpi-listener.hpp>
#include <vector>

#include "./radix_sort.h"

TEST(Parallel_Operations_MPI, Test_incorrect_size) {
  int rank;
  int size = -2;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec;
  if (rank == 0) {
    ASSERT_ANY_THROW(getRandomVector(size));
  }
}

TEST(Parallel_Operation_MPI, Test_work_radix_sort) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec = getRandomVector(15);
  std::vector<int> result(vec.size());
  if (rank == 0) {
    int size = vec.size();
    vec = RadixSort(vec, vec.size());
    result = bubble_sort_Vector(vec);
    for (int i = 0; i < size; ++i) EXPECT_EQ(vec[i], result[i]);
  }
}

TEST(Parallel_Operation_MPI, Test_work_merge) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    std::vector<int> vec1{3, 4, 10, 17};
    std::vector<int> vec2{0, 6, 7, 46};
    std::vector<int> sum{0, 3, 4, 6, 7, 10, 17, 46};
    std::vector<int> result = Merge(vec1, vec2);
    for (int i = 0; i < 8; ++i) EXPECT_EQ(result[i], sum[i]);
  }
}

TEST(Parallel_Operations_MPI, Test_const_ParallelRadixSort) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  std::vector<int> vec{47, 39, 118, 5,   24, 7,  27, 22,  35, 48,
                       70, 56, 107, 461, 19, 32, 95, 206, 67, 11};
  std::vector<int> result = ParallSort(vec);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
    std::vector<int> simple_sort = bubble_sort_Vector(vec);
    for (int i = 0; i < 10; ++i) EXPECT_EQ(result[i], simple_sort[i]);
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
