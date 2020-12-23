// Copyright 2018 Nesterov Alexander

#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./radix_sort_double_simple_merge.h"

using std::vector;
using std::cout;
using std::endl;

TEST(Parallel_Operations_MPI, Test_Negative_Size) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int size = -999;
  if (rank == 0) {
    ASSERT_ANY_THROW(getRandomVector(size));
  }
}

TEST(Parallel_Operations_MPI, Test_Bubble_sort) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  const int size = 100;

  if (rank == 0) {
    auto vec = getRandomVector(size);
    vec = bubbleSort(vec);
    for (int i = 0; i < vec.size()-1; i++) {
      ASSERT_TRUE(vec[i] <= vec[i+1]);
    }
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

TEST(Parallel_Operations_MPI, Test_NumberOfPositiveRadix) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const int number = 142646;
  if (rank == 0) {
    EXPECT_EQ(numberOfPositiveRadix(number), 6);
  }
}

TEST(Parallel_Operations_MPI, Test_NumberOfNegativeRadix) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        double number = 345645.345236;
        auto koll_digit = numberOfNegativeRadix(number);
        EXPECT_EQ(koll_digit, 6);
    }
}

TEST(Parallel_Operations_MPI, TEST_Merge) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0) {
    vector<double> vec1{ 1.23, 1.38, 3.23, 5.56 };
    vector<double> vec2{ 0.23, 2.2, 3.6, 6.23 };
    vector<double> res{ 0.23, 1.23, 1.38, 2.2, 3.23, 3.6, 5.56, 6.23};
    vector<double> vec3 = merge(vec1, vec2);

    for (int i = 0; i < 8; ++i) {
      EXPECT_EQ(vec3[i], res[i]) << "Arrays check_array and res_array differ at index " << i;
    }
  }
}

TEST(Parallel_Operations_MPI, TEST_GetDigit) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double number = 473.5;

    if (rank == 0) {
        int a = getDigit(number, -1);
        int b = getDigit(number, 0);
        int c = getDigit(number, 1);
        int d = getDigit(number, 2);

        EXPECT_EQ(a, 5);
        EXPECT_EQ(b, 3);
        EXPECT_EQ(c, 7);
        EXPECT_EQ(d, 4);

        int e = getDigit(89.12345, -5);
        EXPECT_EQ(e, 5);
  }
}

TEST(Parallel_Operations_MPI, TEST_RadixSort) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        vector<double> vect = { 0.1, 0.6, 0.3, 0.9, 0.2 };
        vector<double> result = { 0.1, 0.2, 0.3, 0.6, 0.9 };
        auto sorted = radixSort(vect, -1);
        EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, Test_LinearRadixSort) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
            vector<double> v = { 56.68, 777.4, 12.14, 999.15243, 30.1, 92.2 };
            vector<double> result = { 12.14, 30.1, 56.68, 92.2, 777.4, 999.15243 };
            vector<double> sorted = linearRadixSort(v);
            EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, Test_ParallelRadixSort) {
    int rank;
    vector<double> v(6);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        vector<double> tmp = { 56.68, 777.4, 12.14, 999.15243, 30.1, 92.2 };
        v = tmp;
    }
    auto sorted = parallelRadixSort(v);
    if (rank == 0) {
        vector<double> result = { 12.14, 30.1, 56.68, 92.2, 777.4, 999.15243 };
        EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, parallel_bitwise_sort_works_rand_vector) {
    int rank;
    double start, end;
    double seconds;

    vector<double> v(60000);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        v = getRandomVector(60000);
    }
    start = MPI_Wtime();
    auto sorted = parallelRadixSort(v);
    end = MPI_Wtime();

    if (rank == 0) {
        seconds = end - start;
        std::cout << "Parallel: " << seconds << std::endl;

        start = MPI_Wtime();
        vector<double> result = linearRadixSort(v);
        end = MPI_Wtime();
        seconds = end - start;
        std::cout << "Linear: " << seconds << std::endl;
        EXPECT_EQ(sorted, result);
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
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  cout << rank << endl;
}
