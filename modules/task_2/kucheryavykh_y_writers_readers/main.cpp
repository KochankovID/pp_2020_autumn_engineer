// Copyright 2020 Kucheryavykh Yan
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include "./writers_readers.h"

TEST(Parallel_Operations_MPUI, Test_task_1) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int writing = 10;
    int reading = 10;

    int variable;
    if (rank == 0) {
        variable = main_process(writing * ((size-1) / 3) + reading * ((size - 1) - ((size-1) / 3)));
    }

    if (rank != 0) {
        if (rank % 3 == 0) {
            writer(writing);
        } else {
            reader(reading);
        }
    }

    if (rank == 0) {
        ASSERT_EQ(variable, writing * ((size-1) / 3));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_2) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int writing = 5;
    int reading = 7;

    int variable;
    if (rank == 0) {
        variable = main_process(writing * ((size-1) / 3) + reading * ((size - 1) - ((size-1) / 3)));
    }

    if (rank != 0) {
        if (rank % 3 == 0) {
            writer(writing);
        } else {
            reader(reading);
        }
    }

    if (rank == 0) {
        ASSERT_EQ(variable, writing * ((size-1) / 3));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_3) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int writing = 5;
    int reading = 2;

    int variable;
    if (rank == 0) {
        variable = main_process(writing * ((size-1) / 3) + reading * ((size - 1) - ((size-1) / 3)));
    }

    if (rank != 0) {
        if (rank % 3 == 0) {
            writer(writing);
        } else {
            reader(reading);
        }
    }

    if (rank == 0) {
        ASSERT_EQ(variable, writing * ((size-1) / 3));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_4) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int writing = 11;
    int reading = 7;

    int variable;
    if (rank == 0) {
        variable = main_process(writing * ((size-1) / 3) + reading * ((size - 1) - ((size-1) / 3)));
    }

    if (rank != 0) {
        if (rank % 3 == 0) {
            writer(writing);
        } else {
            reader(reading);
        }
    }

    if (rank == 0) {
        ASSERT_EQ(variable, writing * ((size-1) / 3));
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_5) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int writing = 7;
    int reading = 11;

    int variable;
    if (rank == 0) {
        variable = main_process(writing * ((size-1) / 3) + reading * ((size - 1) - (size / 3)));
    }

    if (rank != 0) {
        if (rank % 3 == 0) {
            writer(writing);
        } else {
            reader(reading);
        }
    }

    if (rank == 0) {
        ASSERT_EQ(variable, writing * ((size-1) / 3));
    }
    MPI_Barrier(MPI_COMM_WORLD);
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
