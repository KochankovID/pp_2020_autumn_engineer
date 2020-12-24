// Copyright 2020 Kren Polina
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "../../../modules/task_3/kren_p_gaussian_kernel/gaussian_kernel.h"

TEST(Linear_Filtering_MPI, Test_not_correct_size_of_matrix) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        ASSERT_ANY_THROW(get_Img(-1, 3));
    }
}

TEST(Linear_Filtering_MPI, Test_not_correct_solve_1) {
    std::vector <int> res;
    int rank;
    std::vector <int> mask = { 0, 0, 0, 0, 1, 0, 0, 0, 0 };
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        res = get_Img(5, 7);
        ASSERT_ANY_THROW(parall_linear_filter(mask, res, 3, 2));
    }
}

TEST(Linear_Filtering_MPI, Test_correct_solve_2) {
    int rows = 3, columns = 3;
    std::vector <int> mask = { 0, 1, 0, 1, -4, 1, 0, 1, 0 };
    std::vector <int> res(rows * columns), res_seq(rows * columns);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        res[0] = 150;
        res[1] = 128;
        res[2] = 100;
        res[3] = 200;
        res[4] = 175;
        res[5] = 100;
        res[6] = 50;
        res[7] = 250;
        res[8] = 200;
        res_seq[0] = 28;
        res_seq[1] = 41;
        res_seq[2] = 28;
        res_seq[3] = -225;
        res_seq[4] = -22;
        res_seq[5] = 175;
        res_seq[6] = 350;
        res_seq[7] = -325;
        res_seq[8] = -50;
    }
    std::vector <int> ans(rows * columns);
    ans = parall_linear_filter(mask, res, rows, columns);
    if (rank == 0) {
        ASSERT_EQ(res_seq, ans);
    }
}

TEST(Linear_Filtering_MPI, Test_correct_solve_3) {
    int rows = 10, columns = 10;
    std::vector <int> mask = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    std::vector <int> result(rows * columns);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        result = get_Img(rows, columns);
    }
    std::vector <int> ans(rows * columns);
    ans = parall_linear_filter(mask, result, rows, columns);
    std::vector <int> ans_seq(rows * columns);
    if (rank == 0) {
        ans_seq = SequentialLinearFilter(mask, result, rows, columns);
        ASSERT_EQ(ans_seq, ans);
    }
}

TEST(Linear_Filtering_MPI, Test_correct_solve_4) {
    int rows = 20, cols = 40;
    std::vector <int> mask = { 1, 2, 1, 2, 4, 2, 1, 2, 1 };
    std::vector <int> result(rows * cols);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        result = get_Img(rows, cols);
    }
    std::vector <int> ans(rows * cols);
    ans = parall_linear_filter(mask, result, rows, cols);
    std::vector <int> ans_seq(rows * cols);
    if (rank == 0) {
        ans_seq = SequentialLinearFilter(mask, result, rows, cols);
        ASSERT_EQ(ans_seq, ans);
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
