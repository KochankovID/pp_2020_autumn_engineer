// Copyright 2020 Voronin Aleksey
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./multi_scalar.h"

TEST(Parallel_Operations_MPI, Test_Scalar) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vec, second_global_vec;
    const int count_size_vector = 100;

    if (rank == 0) {
        global_vec = getRandomVector(count_size_vector);
        second_global_vec = getRandomVector(count_size_vector);
    }

    int global_sum = getParallelOperations(global_vec, second_global_vec, count_size_vector);

    if (rank == 0) {
        int reference_sum = getSequentialOperations(global_vec, second_global_vec);
        ASSERT_EQ(global_sum, reference_sum);
    }
}

TEST(Parallel_Operations_MPI, Test_Cant_Get_Vector_With_Wrong_Size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vec, second_global_vec;
    const int count_size_vector = -100;

    if (rank == 0) {
        ASSERT_ANY_THROW(getRandomVector(count_size_vector));
    }
}

TEST(Parallel_Operations_MPI, Test_Can_Create_Vector) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vec, second_global_vec;
    const int count_size_vector = 100;

    if (rank == 0) {
        ASSERT_NO_THROW(global_vec = getRandomVector(count_size_vector));
    }
}

TEST(Parallel_Operations_MPI, Test_Can_Get_Parallel_Operations) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vec, second_global_vec;
    const int count_size_vector = 100;

    if (rank == 0) {
        global_vec = getRandomVector(count_size_vector);
        second_global_vec = getRandomVector(count_size_vector);
    }

    ASSERT_NO_THROW(getParallelOperations(global_vec, second_global_vec, count_size_vector));
}

TEST(Parallel_Operations_MPI, Test_Can_Get_Sequential_Operations) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<int> global_vec, second_global_vec;
    const int count_size_vector = 100;

    if (rank == 0) {
        global_vec = getRandomVector(count_size_vector);
        second_global_vec = getRandomVector(count_size_vector);
        ASSERT_NO_THROW(getSequentialOperations(global_vec, second_global_vec));
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
