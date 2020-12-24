// Copyright 2020 Voronin Aleksey
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include "./batcher_quick_sort.h"

TEST(Parallel_Operations_MPI, Test_Can_Create_Vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::vector<int> vector;
        const int size = 100;

        ASSERT_NO_THROW(vector = getRandomVector(size));
    }
}

TEST(Parallel_Operations_MPI, Test_Can_Quick_Sort_Vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        const int size = 100;
        std::vector<int> vector = getRandomVector(size);

        ASSERT_NO_THROW(quickSort(vector));
    }
}

TEST(Parallel_Operations_MPI, Test_Quick_Sort_Vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::vector<int> vector = {0, 4, 3, 8, 9, 5, 27, 3};
        std::vector<int> result = {0, 3, 3, 4, 5, 8, 9, 27};
        vector = quickSort(vector);
        for (size_t i = 0; i < vector.size(); i++) {
            EXPECT_EQ(result[i], vector[i]);
        }
    }
}

TEST(Parallel_Operations_MPI, Test_Parallel_Sort_Vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> vector = {1, 4, 3, 8, 9, 5, 27, 2, 10, 12, 11, 46, 23, 45, 47, 48, 49};
    std::vector<int> result = {1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 23, 27, 45, 46, 47, 48, 49};
    std::vector<int> realresult = parallelSort(vector);
    if (rank == 0) {
        for (size_t i = 0; i < vector.size(); i++) {
            EXPECT_EQ(result[i], realresult[i]);
        }
    }
}

TEST(Parallel_Operations_MPI, Test_Parallel_Sort_Random_Vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> vector = getRandomVector(128);
    std::vector<int> result = parallelSort(vector);
    if (rank == 0) {
        for (size_t i = 0; i < vector.size() - 1; i++) {
            EXPECT_TRUE(result[i] <= result[i + 1]);
        }
    }
}

TEST(Parallel_Operations_MPI, Test_Parallel_Sort_Big_Random_Vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> vector = getRandomVector(100);
    std::vector<int> result = parallelSort(vector);
    if (rank == 0) {
        for (size_t i = 0; i < vector.size() - 1; i++) {
            EXPECT_TRUE(result[i] <= result[i + 1]);
        }
    }
}

TEST(Parallel_Operations_MPI, Test_Time) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> vector = getRandomVector(1000);
    double p_beg, p_end, s_beg, s_end;
    p_beg = MPI_Wtime();
    std::vector<int> result = parallelSort(vector);
    p_end = MPI_Wtime();
    if (rank == 0) {
        s_beg = MPI_Wtime();
        vector = quickSort(vector);
        s_end = MPI_Wtime();
        ASSERT_EQ(result, vector);
        std::cout << "Parallel time= " << p_end - p_beg << std::endl
                  << "Sequential time= " << s_end - s_beg << std::endl;
    }
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners &listeners =
        ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
