// Copyright 2020 Eremina Alena
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "./cannon.h"

TEST(Cannon_Algorithm_MPI, Test_Get_Rangom_Matrix) {
    int size = 6;
    std::vector<double> a(size*size);
    ASSERT_NO_THROW(a = getRandomMatrix(size));
}

TEST(Cannon_Algorithm_MPI, Test_Sequential_Multiplication) {
    int ProcNum, ProcRank;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    int size = 3;

    std::vector<double> a(size*size);
    std::vector<double> b(size*size);
    std::vector<double> c_ref(size*size);
    std::vector<double> c(size*size, 0.0);
    a = { 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9 };
    b = { 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9 };
    c_ref = { 0.9, 0.96, 1.02, 2.16, 2.31, 2.46, 3.42, 3.66, 3.9 };
    if (ProcRank == 0) {
        c = getSequentialMultiply(a, b, size);
        int count = 0;
        for (int i = 0; i < size * size; i++) {
            if (abs(c[i] - c_ref[i]) > 0.000001) count++;
        }
        ASSERT_EQ(0, count);
    }
}

TEST(Cannon_Algorithm_MPI, Test_Parallel_Multiplication) {
    int ProcNum, ProcRank;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    int size = 440 * static_cast<int>(sqrt(ProcNum));
    std::vector<double> a(size*size);
    std::vector<double> b(size*size);
    std::vector<double> c_seq(size*size, 0.0);
    std::vector<double> c_par(size*size, 0.0);
    a = getRandomMatrix(size);
    b = getRandomMatrix(size);
    double t0 = MPI_Wtime();
    c_par = getParallelMultiply(a, b, size);
    double t1 = MPI_Wtime();
    if (ProcRank == 0) {
        printf("Par: %f\n", t1 - t0);
        t0 = MPI_Wtime();
        c_seq = getSequentialMultiply(a, b, size);
        t1 = MPI_Wtime();
        printf("Seq: %f\n", t1 - t0);
        int count = 0;
        for (int i = 0; i < size*size; i++) {
            if (abs(c_seq[i] - c_par[i]) > 0.000001) count++;
        }
        ASSERT_EQ(0, count);
    }
}

TEST(Cannon_Algorithm_MPI, Test_Parallel_Multiplication_With_One_Elems_In_Block) {
    int ProcNum, ProcRank;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    int size = static_cast<int>(sqrt(ProcNum));
    std::vector<double> a(size*size);
    std::vector<double> b(size*size);
    std::vector<double> c_seq(size*size, 0.0);
    std::vector<double> c_par(size*size, 0.0);
    a = getRandomMatrix(size);
    b = getRandomMatrix(size);
    c_par = getParallelMultiply(a, b, size);
    if (ProcRank == 0) {
        c_seq = getSequentialMultiply(a, b, size);
        int count = 0;
        for (int i = 0; i < size*size; i++) {
            if (abs(c_seq[i] - c_par[i]) > 0.000001) count++;
        }
        ASSERT_EQ(0, count);
    }
}

TEST(Cannon_Algorithm_MPI, Test_Parallel_Mitliplication_With_Wrong_Size) {
    int ProcNum, ProcRank;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    int size = 99;
    std::vector<double> a(size*size);
    std::vector<double> b(size*size);
    std::vector<double> c_par(size*size, 0.0);
    a = getRandomMatrix(size);
    b = getRandomMatrix(size);
    ASSERT_NO_THROW(c_par = getParallelMultiply(a, b, size));
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
