// Copyright 2020 Ekaterina Burmistrova
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <math.h>
#include <algorithm>
#include <vector>

#include "./shell_sort.h"

TEST(Parallel_sort, sort_random_vect) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    std::vector<int> vect = { 15, -17, 18, 8, -14, -10, 16, 12, -6, -9 };
    std::vector<int> res = { -17, -14, -10, -9, -6, 8, 12, 15, 16, 18 };

    std::vector<int> sortArray = Parallel_sort(vect);

    if (mynode == 0) {
        ASSERT_EQ(sortArray, res);
    }
}

TEST(Parallel_sort, sort_ascending_vect) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    std::vector<int> vect = { 9, 8, 7, 6, 5, 4 };
    std::vector<int> res = { 4, 5, 6, 7, 8, 9 };

    std::vector<int> sortArray = Parallel_sort(vect);

    if (mynode == 0) {
        ASSERT_EQ(sortArray, res);
    }
}

TEST(Parallel_sort, sort_decreasing_vect) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    std::vector<int> vect = { 1, 2, 3, 4, 5, 6, 7 };
    std::vector<int> res = { 1, 2, 3, 4, 5, 6, 7 };

    std::vector<int> sortArray = Parallel_sort(vect);

    if (mynode == 0) {
        ASSERT_EQ(sortArray, res);
    }
}

TEST(Parallel_sort, sort_vect_with_one_element) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    std::vector<int> vect = { 7 };
    std::vector<int> res = { 7 };

    std::vector<int> sortArray = Parallel_sort(vect);

    if (mynode == 0) {
        ASSERT_EQ(sortArray, res);
    }
}

TEST(Parallel_sort, return_empty_vect) {
    int mynode;
    MPI_Comm_rank(MPI_COMM_WORLD, &mynode);

    std::vector<int> vect = { };
    std::vector<int> res = { };

    std::vector<int> sortArray = Parallel_sort(vect);

    if (mynode == 0) {
        ASSERT_EQ(sortArray, res);
    }
}


int main(int argc, char **argv) {
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
