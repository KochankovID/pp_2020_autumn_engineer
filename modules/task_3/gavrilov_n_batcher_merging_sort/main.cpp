// Copyright 2020 Gavrilov Nikita
#define _SILENCE_TR1_NAMESPACE_DEPRECATION_WARNING

#include <stdio.h>
#include <mpi.h>
#include <gtest/gtest.h>
#include <gtest-mpi-listener.hpp>
#include "BatcherMergingSort.h"
#include <vector>
#include <iostream>
#include <ctime>
#include <random>

bool IsSorted(const std::vector<int>& data) {
    if (data.size() == 0)
        return true;
    for (size_t i = 0; i < data.size()-1; i++) {
        if (data[i] > data[i + 1])
            return false;
    }
    return true;
}
std::vector<int> GetRandomVector(size_t size) {
    std::mt19937 rnd;
    rnd.seed(static_cast<unsigned int>(time(0)));

    std::vector<int> result(size);
    for (size_t i = 0; i < size; i++) {
        result[i] = rnd() % 20000 - 10000;
    }

    return result;
}

TEST(Parallel_Operations_MPI, No_Throw) {
    std::vector<int> data = GetRandomVector(10000);
    ASSERT_NO_THROW(Sort(&data));
}

TEST(Parallel_Operations_MPI, Sorting_Right) {
    std::vector<int> data = GetRandomVector(10000);
    Sort(&data);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_TRUE(IsSorted(data));
    }
}

TEST(Parallel_Operations_MPI, Sort_By_Num_Place_Rang10) {
    std::vector<int> data = GetRandomVector(10000);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_NO_THROW(SortByNumPlace(&data, 10));
        ASSERT_TRUE(IsSorted(data));
    }
}

TEST(Parallel_Operations_MPI, Sort_By_Num_Place_Rang2) {
    std::vector<int> data = GetRandomVector(10000);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_NO_THROW(SortByNumPlace(&data, 2));
        ASSERT_TRUE(IsSorted(data));
    }
}

TEST(Parallel_Operations_MPI, Sorting_Right_Even_If_Bad_Size) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int i = 1; i < size; i++) {
        std::vector<int> data = GetRandomVector(size * 1000 + size - i);
        ASSERT_NO_THROW(Sort(&data));
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            ASSERT_TRUE(IsSorted(data));
        }
    }
}

TEST(Parallel_Operations_MPI, Sorting_Right_Even_If_Bad_Size2) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (int i = 1; i <= size; i++) {
        std::vector<int> data = GetRandomVector(size - 1);
        ASSERT_NO_THROW(Sort(&data));
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            ASSERT_TRUE(IsSorted(data));
        }
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
