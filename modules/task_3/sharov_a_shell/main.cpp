// Copyright 2018 Sharov Alexander
#include <mpi.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>
#include "./shell.h"


TEST(Parallel_Sort, Test1) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 2500;
    std::vector<int> seqres;
    std::vector<int> parallelres;
    if (rank == 0) {
        seqres = genvector(size);
        parallelres = seqres;
    }
    parallelres = parallelsortshell(parallelres, size);
    if (rank == 0) {
        double t1 = MPI_Wtime();
        seqres = sortshell(seqres, size);
        double t2 = MPI_Wtime();
        std::cout << "seq time: " << t2 - t1 << std::endl;
        ASSERT_EQ(seqres, parallelres);
    }
}

TEST(Parallel_Sort, Test2) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 5609;
    std::vector<int> seqres;
    std::vector<int> parallelres;
    if (rank == 0) {
        seqres = genvector(size);
        parallelres = seqres;
    }
    parallelres = parallelsortshell(parallelres, size);
    if (rank == 0) {
        double t1 = MPI_Wtime();
        seqres = sortshell(seqres, size);
        double t2 = MPI_Wtime();
        std::cout << "seq time: " << t2 - t1 << std::endl;
        ASSERT_EQ(seqres, parallelres);
    }
}

TEST(Parallel_Sort, Test3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 8888;
    std::vector<int> seqres;
    std::vector<int> parallelres;
    if (rank == 0) {
        seqres = genvector(size);
        parallelres = seqres;
    }
    parallelres = parallelsortshell(parallelres, size);
    if (rank == 0) {
        double t1 = MPI_Wtime();
        seqres = sortshell(seqres, size);
        double t2 = MPI_Wtime();
        std::cout << "seq time: " << t2 - t1 << std::endl;
        ASSERT_EQ(seqres, parallelres);
    }
}

TEST(Parallel_Sort, Test4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 10000;
    std::vector<int> seqres;
    std::vector<int> parallelres;
    if (rank == 0) {
        seqres = genvector(size);
        parallelres = seqres;
    }
    parallelres = parallelsortshell(parallelres, size);
    if (rank == 0) {
        double t1 = MPI_Wtime();
        seqres = sortshell(seqres, size);
        double t2 = MPI_Wtime();
        std::cout << "seq time: " << t2 - t1 << std::endl;
        ASSERT_EQ(seqres, parallelres);
    }
}

TEST(Parallel_Sort, Test5) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int size = 50123;
    std::vector<int> seqres;
    std::vector<int> parallelres;
    if (rank == 0) {
        seqres = genvector(size);
        parallelres = seqres;
    }
    parallelres = parallelsortshell(parallelres, size);
    if (rank == 0) {
        double t1 = MPI_Wtime();
        seqres = sortshell(seqres, size);
        double t2 = MPI_Wtime();
        std::cout << "seq time: " << t2 - t1 << std::endl;
        ASSERT_EQ(seqres, parallelres);
    }
}


int main(int argc, char* argv[]) {
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

