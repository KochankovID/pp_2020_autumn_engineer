// Copyright 2020 Chesnokov
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include "./Strassen.h"

using std::cout;
using std::endl;

TEST(Task_3, Test_Sequential_Mul_Works) {
    double data_a[] = { 1, 2, 1,  0, 3, 2 };
    double data_b[] = { 2, 3,  0, 2,  1, 0 };
    double res[] = { 3, 7,  2, 6 };

    Matrix a(3, 2, data_a);
    Matrix b(2, 3, data_b);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        Matrix seq_res = getMatrixMul(a, b);
        for (int i = 0; i < seq_res.rows; i++)
            for (int j = 0; j < seq_res.columns; j++) {
                EXPECT_EQ(seq_res.buf[i * seq_res.columns + j], res[i * seq_res.columns + j]);
            }
    }
}

TEST(Task_3, Test_Summing_Works) {
    double data_a[] = { 1, 2, 1,  0, 3, 2 };
    double data_b[] = { 2, 3, 0,  2, 1, 0 };
    double res[] = { 3, 5, 1,  2, 4, 2 };

    Matrix a(3, 2, data_a);
    Matrix b(3, 2, data_b);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        Matrix seq_res = a + b;
        for (int i = 0; i < seq_res.rows; i++)
            for (int j = 0; j < seq_res.columns; j++) {
                EXPECT_EQ(seq_res.buf[i * seq_res.columns + j], res[i * seq_res.columns + j]);
            }
    }
}

TEST(Task_3, Test_Sequential_Strassen_Works_256x256) {
    Matrix a = getRandomMatrix(256, 256);
    Matrix b = getRandomMatrix(256, 256);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        Matrix seq_res = getMatrixMul(a, b);
        Matrix par_res = getMatrixMulStrassen(a, b);
        for (int i = 0; i < seq_res.rows; i++)
            for (int j = 0; j < seq_res.columns; j++) {
                EXPECT_EQ(seq_res.buf[i * seq_res.columns + j], par_res.buf[i * seq_res.columns + j]);
            }
    }
}

TEST(Task_3, Test_Sequential_Strassen_Works_512x512) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        Matrix a = getRandomMatrix(512, 512);
        Matrix b = getRandomMatrix(512, 512);
        double seq_start = MPI_Wtime();
        Matrix seq_res = getMatrixMul(a, b);
        double seq_end = MPI_Wtime();
        Matrix str_res = getMatrixMulStrassen(a, b);
        double str_end = MPI_Wtime();
        for (int i = 0; i < seq_res.rows; i++)
            for (int j = 0; j < seq_res.columns; j++) {
                EXPECT_EQ(seq_res.buf[i * seq_res.columns + j], str_res.buf[i * seq_res.columns + j]);
            }
        cout << "Common mult: " << seq_end - seq_start << endl;
        cout << "Strassen mult: " << str_end - seq_end << endl;
    }
}

TEST(Task_3, Test_Parallel_Strassen_Works_1024x1024) {
    int rank;
    int s = 1024;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix a;
    Matrix b;
    Matrix str_res(1, 1);
    double str_start;
    double str_end;
    if (rank == 0) {
        a = getRandomMatrix(s, s);
        b = getRandomMatrix(s, s);
        str_start = MPI_Wtime();
        str_res = getMatrixMulStrassen(a, b);
        str_end = MPI_Wtime();
    }
    double par_start = MPI_Wtime();
    Matrix par_res = getParallelMatrixMul(a, b);
    double par_end = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < str_res.rows; i++)
            for (int j = 0; j < str_res.columns; j++) {
                EXPECT_EQ(str_res.buf[i * str_res.columns + j], par_res.buf[i * par_res.columns + j]);
            }
        cout << "Strassen seq mult: " << str_end - str_start << endl;
        cout << "Strassen parallel mult: " << par_end - par_start << endl;
    }
}

TEST(Task_3, Test_Parallel_Strassen_Works_With_Not_Square_Matrix_1024x512) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix a;
    Matrix b;
    Matrix str_res(1, 1);
    double str_start;
    double str_end;
    if (rank == 0) {
        a = getRandomMatrix(1024, 512);
        b = getRandomMatrix(512, 1024);
        str_start = MPI_Wtime();
        str_res = getMatrixMulStrassen(a, b);
        str_end = MPI_Wtime();
    }
    double par_start = MPI_Wtime();
    Matrix par_res = getParallelMatrixMul(a, b);
    double par_end = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < str_res.rows; i++)
            for (int j = 0; j < str_res.columns; j++) {
                EXPECT_EQ(str_res.buf[i * str_res.columns + j], par_res.buf[i * par_res.columns + j]);
            }
        cout << "Strassen seq mult: " << str_end - str_start << endl;
        cout << "Strassen parallel mult: " << par_end - par_start << endl;
    }
}

TEST(Task_3, Test_Parallel_Strassen_Works_With_Any_Square_Dimension_789x789) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix a;
    Matrix b;
    Matrix str_res(1, 1);
    double str_start;
    double str_end;
    if (rank == 0) {
        a = getRandomMatrix(789, 789);
        b = getRandomMatrix(789, 789);
        str_start = MPI_Wtime();
        str_res = getMatrixMulStrassen(a, b);
        str_end = MPI_Wtime();
    }
    double par_start = MPI_Wtime();
    Matrix par_res = getParallelMatrixMul(a, b);
    double par_end = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < str_res.rows; i++)
            for (int j = 0; j < str_res.columns; j++) {
                EXPECT_EQ(str_res.buf[i * str_res.columns + j], par_res.buf[i * par_res.columns + j]);
            }
        cout << "Strassen seq mult: " << str_end - str_start << endl;
        cout << "Strassen parallel mult: " << par_end - par_start << endl;
    }
}

TEST(Task_3, Test_Parallel_Strassen_Works_With_Any_NOT_Square_Dimension_789x567_922x789) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix a;
    Matrix b;
    Matrix str_res(1, 1);
    double str_start;
    double str_end;
    if (rank == 0) {
        a = getRandomMatrix(789, 567);
        b = getRandomMatrix(922, 789);
        str_start = MPI_Wtime();
        str_res = getMatrixMulStrassen(a, b);
        str_end = MPI_Wtime();
    }
    double par_start = MPI_Wtime();
    Matrix par_res = getParallelMatrixMul(a, b);
    double par_end = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < str_res.rows; i++)
            for (int j = 0; j < str_res.columns; j++) {
                EXPECT_EQ(str_res.buf[i * str_res.columns + j], par_res.buf[i * par_res.columns + j]);
            }
        cout << "Strassen seq mult: " << str_end - str_start << endl;
        cout << "Strassen parallel mult: " << par_end - par_start << endl;
    }
}
/*
TEST(Task_3, Test_Time_With_Big_Matrix_2048x2048) {
    int rank;
    int s = 2048;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Matrix a;
    Matrix b;
    Matrix str_res(1, 1);
    double str_start;
    double str_end;
    if (rank == 0) {
        a = getRandomMatrix(s, s);
        b = getRandomMatrix(s, s);
        str_start = MPI_Wtime();
        str_res = getMatrixMulStrassen(a, b);
        str_end = MPI_Wtime();
    }
    double par_start = MPI_Wtime();
    Matrix par_res = getParallelMatrixMul(a, b);
    double par_end = MPI_Wtime();
    if (rank == 0) {
        for (int i = 0; i < str_res.rows; i++)
            for (int j = 0; j < str_res.columns; j++) {
                EXPECT_EQ(str_res.buf[i * str_res.columns + j], par_res.buf[i * par_res.columns + j]);
            }
        cout << "Strassen seq mult: " << str_end - str_start << endl;
        cout << "Strassen parallel mult: " << par_end - par_start << endl;
    }
}
*/
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
