// Copyright 2020 Kryukov Sergey
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include "./opersobel.h"

TEST(sobel_operator, mat3x4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int row = 3, col = 4;
    std::vector<int> paral_res(row * col);
    std::vector<int> mat(row * col);
    for (int i = 0; i < row * col; i++) {
        mat[i] = 11;
    }
    paral_res = paral_sobel_oper(mat, row, col);
    if (rank == 0) {
        std::vector<int> res(row * col);
        res = sobel_for_oneproc(mat, row, col);
        ASSERT_EQ(res, paral_res);
    }
}

TEST(sobel_operator, mat6x6) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int row = 6, col = 6;
    std::vector<int> paral_res(row * col);
    std::vector<int> mat(row * col);
    for (int i = 0; i < row * col; i++) {
        mat[i] = 254;
    }
    paral_res = paral_sobel_oper(mat, row, col);
    if (rank == 0) {
        std::vector<int> res(row * col);
        res = sobel_for_oneproc(mat, row, col);
        ASSERT_EQ(res, paral_res);
    }
}

TEST(sobel_operator, mat5x5) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int row = 5, col = 5;
    std::vector<int> paral_res(row * col);
    std::vector<int> mat(row * col);
    for (int i = 0; i < row * col; i++) {
        mat[i] = 132;
    }
    paral_res = paral_sobel_oper(mat, row, col);
    if (rank == 0) {
        std::vector<int> res(row * col);
        res = sobel_for_oneproc(mat, row, col);
        ASSERT_EQ(res, paral_res);
    }
}

TEST(sobel_operator, mat4x4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int row = 4, col = 4;
    std::vector<int> paral_res(row * col);
    std::vector<int> mat(row * col);
    for (int i = 0; i < row * col; i++) {
        mat[i] = 43;
    }
    paral_res = paral_sobel_oper(mat, row, col);
    if (rank == 0) {
        std::vector<int> res(row * col);
        res = sobel_for_oneproc(mat, row, col);
        ASSERT_EQ(res, paral_res);
    }
}

TEST(sobel_operator, mat3x3) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int row = 3, col = 3;
    std::vector<int> paral_res(row * col);
    std::vector<int> mat(row * col);
    mat[0] = 32;
    mat[1] = 202;
    mat[2] = 65;
    mat[3] = 222;
    mat[4] = 27;
    mat[5] = 34;
    mat[6] = 243;
    mat[7] = 33;
    mat[8] = 12;
    paral_res = paral_sobel_oper(mat, row, col);
    if (rank == 0) {
        std::vector<int> res(row * col);
        res = sobel_for_oneproc(mat, row, col);
        ASSERT_EQ(res, paral_res);
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
