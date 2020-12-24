//// Copyright 2020 Egorov Kirill
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./batcher_sort.h"

TEST(basic_functions_tests, shuffle_test) {
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        std::vector<int> testVec = { 10, 20, 60, 35, 47, 23, 12, 30 };
        std::vector<int> checkVec = { 10, 47, 20, 23, 60, 12, 35, 30 };

        sh(&testVec, 0, testVec.size());

        for (unsigned int i = 0; i < testVec.size(); i++) {
            ASSERT_EQ(testVec[i], checkVec[i]);
        }
    }
}

TEST(basic_functions_tests, unshuffle_test) {
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        std::vector<int> testVec = { 10, 20, 60, 35, 47, 23, 12, 30 };
        std::vector<int> checkVec = { 10, 60, 47, 12, 20, 35, 23, 30 };

        uns(&testVec, 0, testVec.size());

        for (unsigned int i = 0; i < testVec.size(); i++) {
            ASSERT_EQ(testVec[i], checkVec[i]);
        }
    }
}

TEST(basic_functions_test, shuffle_unshuffle_is_correctly_reversed_test) {
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        std::vector<int> testVec = { 10, 20, 60, 35, 47, 23, 12, 30 };
        std::vector<int> checkVec = { 10, 20, 60, 35, 47, 23, 12, 30 };

        sh(&testVec, 0, testVec.size());
        uns(&testVec, 0, testVec.size());

        for (unsigned int i = 0; i < testVec.size(); i++) {
            ASSERT_EQ(testVec[i], checkVec[i]);
        }
    }
}

TEST(basic_functions_test, unshuffle_shuffle_is_correctly_reversed_test) {
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        std::vector<int> testVec = { 10, 20, 60, 35, 47, 23, 12, 30 };
        std::vector<int> checkVec = { 10, 20, 60, 35, 47, 23, 12, 30 };

        uns(&testVec, 0, testVec.size());
        sh(&testVec, 0, testVec.size());

        for (unsigned int i = 0; i < testVec.size(); i++) {
            ASSERT_EQ(testVec[i], checkVec[i]);
        }
    }
}

TEST(basic_functions_test, simple_sort_test) {
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        std::vector<int> testVec = getRandomVector(8);

        simpleSort(&testVec);

        ASSERT_EQ(true, simpleCheck(testVec));
    }
}

TEST(sequential_operation_tests, basic_BatcherSort_test) {
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        std::vector<int> testVec = getRandomVector(8);

        BatcherSort(&testVec);

        ASSERT_EQ(true, simpleCheck(testVec));
    }
}

TEST(sequential_operation_tests, basic_BatcherSort_test_large) {
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        std::vector<int> testVec = getRandomVector(8192);

        BatcherSort(&testVec);

        ASSERT_EQ(true, simpleCheck(testVec));
    }
}

TEST(sequential_operation_tests, basic_BatcherSort_test_non_grade_of_two) {
    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    if (rank == 0) {
        std::vector<int> testVec = getRandomVector(12);

        BatcherSort(&testVec);

        ASSERT_EQ(true, simpleCheck(testVec));
    }
}

TEST(parallel_operation_test, batcher_sort_parallel_test) {
    std::vector<int> testVec = getRandomVector(40);

    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    BatcherSortParallel(&testVec);

    if (rank == 0) {
       ASSERT_EQ(true, simpleCheck(testVec));
    }
}

TEST(parallel_operation_test, batcher_sort_parallel_test_large) {
    std::vector<int> testVec = getRandomVector(8192);

    int rank, numtasks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    BatcherSortParallel(&testVec);

    if (rank == 0) {
       ASSERT_EQ(true, simpleCheck(testVec));
    }
}

TEST(efficiency_test, parallel_sequential_times_test_large) {
    std::vector<int> testVec = getRandomVector(8192);

    int rank, numtasks;
    double starttime, endtime, dif_parallel_time;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);

    starttime = MPI_Wtime();
    BatcherSortParallel(&testVec);
    endtime = MPI_Wtime();
    dif_parallel_time = endtime - starttime;

    if (rank == 0) {
        clock_t start, end;
        double dif_sequential_time;

        start = clock();
        BatcherSort(&testVec);
        end = clock();

        dif_sequential_time = static_cast<double>(end - start);

        std::cout << "Parallel time " << dif_parallel_time << std::endl;
        std::cout << "Sequential time " << dif_sequential_time << std::endl;

        ASSERT_TRUE(dif_sequential_time >= dif_parallel_time);
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
