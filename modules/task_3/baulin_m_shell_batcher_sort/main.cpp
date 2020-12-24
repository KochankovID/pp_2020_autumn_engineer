// Copyright 2020 Mikhail Baulin
#include <gtest/gtest.h>
#include <vector>
#include <numeric>
#include <algorithm>
#include "./baulin_m_shell_batcher_sort.h"
#include "../../../3rdparty/gtest-mpi/linux/gtest-mpi-listener.hpp"


TEST(Parallel_shell_batcher_vectort_sort_MPI, Test_1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<int> vec(1);

    if (rank == 0) {
        vec = std::vector<int>{ 8, 7, 2, 1, 4, 3, 6, 5 };
    }

    std::vector<int> global_result;
    global_result = getParallelOperations(vec);

    if (rank == 0) {
        std::vector<int> reference_result(vec);
        sort(reference_result.begin(), reference_result.end());
        ASSERT_EQ(reference_result, global_result);
    }
}

TEST(Parallel_shell_batcher_vectort_sort_MPI, Test_2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rand_vec_size = 1000;
    int rand_limit = 1000;

    std::vector<int> vec(1);
    if (rank == 0) {
        vec = getVectorWithRandomNumbers(rand_vec_size, rand_limit);
    }

    std::vector<int> global_result;
    global_result = getParallelOperations(vec);

    if (rank == 0) {
        std::vector<int> reference_result(vec);
        sort(reference_result.begin(), reference_result.end());
        ASSERT_EQ(reference_result, global_result);
    }
}

TEST(Parallel_shell_batcher_vectort_sort_MPI, Test_3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rand_vec_size = 1000000;
    int rand_limit = 1000;

    std::vector<int> vec(1);
    if (rank == 0) {
        vec = getVectorWithRandomNumbers(rand_vec_size, rand_limit);
    }

    std::vector<int> global_result;
    global_result = getParallelOperations(vec);

    if (rank == 0) {
        std::vector<int> reference_result(vec);
        sort(reference_result.begin(), reference_result.end());
        ASSERT_EQ(reference_result, global_result);
    }
}

TEST(Parallel_shell_batcher_vectort_sort_MPI, Test_4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rand_vec_size = 1000;
    int rand_limit = 1000000;

    std::vector<int> vec(1);
    if (rank == 0) {
        vec = getVectorWithRandomNumbers(rand_vec_size, rand_limit);
    }

    std::vector<int> global_result;
    global_result = getParallelOperations(vec);

    if (rank == 0) {
        std::vector<int> reference_result(vec);
        sort(reference_result.begin(), reference_result.end());
        ASSERT_EQ(reference_result, global_result);
    }
}

TEST(Parallel_shell_batcher_vectort_sort_MPI, Test_5) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int rand_vec_size = 1000000;
    int rand_limit = 1000000;

    std::vector<int> vec(1);
    if (rank == 0) {
        vec = getVectorWithRandomNumbers(rand_vec_size, rand_limit);
    }

    std::vector<int> global_result;
    global_result = getParallelOperations(vec);

    if (rank == 0) {
        std::vector<int> reference_result(vec);
        sort(reference_result.begin(), reference_result.end());
        ASSERT_EQ(reference_result, global_result);
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
