// Copyright 2020 Kamskov Evgeny
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "./kamskov_e_grackham_hull.h"

TEST(Parallel_Operations_MPI, Test_task_10) {
    int proc_size, proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    std::vector<Point> vector, vector1, vector2;
    int Dimension = 10, points_size = 35;
    double time;

    if (proc == 0) {
        std::cout << "Generating field..." << std::endl;
        vector = get_field(Dimension, points_size);
        print_field(vector, Dimension);
        time = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    vector1 = greckham_par(vector, points_size);

    if (proc == 0) {
        std::cout << "Grackham parallel:" << MPI_Wtime() - time << std::endl;
        time = MPI_Wtime();
        vector2 = greckham_seq(vector);
        std::cout << "Grackham sequential:" << MPI_Wtime() - time << std::endl;
        std::cout << "Convex hull points:" << std::endl;
        for (size_t i = 0; i < vector2.size(); i++) {
            std::cout << "(" << vector2[i].x << ", " << vector2[i].y << ")" << std::endl;
        }
        ASSERT_EQ(vector1, vector2);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_15) {
    int proc_size, proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    std::vector<Point> vector, vector1, vector2;
    int Dimension = 15, points_size = 100;
    double time;

    if (proc == 0) {
        std::cout << "Generating field..." << std::endl;
        vector = get_field(Dimension, points_size);
        print_field(vector, Dimension);
        time = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    vector1 = greckham_par(vector, points_size);

    if (proc == 0) {
        std::cout << "Grackham parallel:" << MPI_Wtime() - time << std::endl;
        time = MPI_Wtime();
        vector2 = greckham_seq(vector);
        std::cout << "Grackham sequential:" << MPI_Wtime() - time << std::endl;
        std::cout << "Convex hull points:" << std::endl;
        for (size_t i = 0; i < vector2.size(); i++) {
            std::cout << "(" << vector2[i].x << ", " << vector2[i].y << ")" << std::endl;
        }
        ASSERT_EQ(vector1, vector2);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_10000) {
    int proc_size, proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    std::vector<Point> vector, vector1, vector2;
    int Dimension = 10000, points_size = 10000;
    double time;

    if (proc == 0) {
        std::cout << "Generating field..." << std::endl;
        vector = get_field(Dimension, points_size);
        // print_field(vector, Dimension);
        time = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    vector1 = greckham_par(vector, points_size);

    if (proc == 0) {
        std::cout << "Grackham parallel:" << MPI_Wtime() - time << std::endl;
        time = MPI_Wtime();
        vector2 = greckham_seq(vector);
        std::cout << "Grackham sequential:" << MPI_Wtime() - time << std::endl;
        ASSERT_EQ(vector1, vector2);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_50000) {
    int proc_size, proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    std::vector<Point> vector, vector1, vector2;
    int Dimension = 8000, points_size = 30000;
    double time;

    if (proc == 0) {
        std::cout << "Generating field..." << std::endl;
        vector = get_field(Dimension, points_size);
        // print_field(vector, Dimension);
        time = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    vector1 = greckham_par(vector, points_size);

    if (proc == 0) {
        std::cout << "Grackham parallel:" << MPI_Wtime() - time << std::endl;
        time = MPI_Wtime();
        vector2 = greckham_seq(vector);
        std::cout << "Grackham sequential:" << MPI_Wtime() - time << std::endl;
        ASSERT_EQ(vector1, vector2);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_100000) {
    int proc_size, proc;
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    std::vector<Point> vector, vector1, vector2;
    int Dimension = 800000, points_size = 100000;
    double time;

    if (proc == 0) {
        std::cout << "Generating field..." << std::endl;
        vector = get_field(Dimension, points_size);
        // print_field(vector, Dimension);
        time = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    vector1 = greckham_par(vector, points_size);

    if (proc == 0) {
        std::cout << "Grackham parallel:" << MPI_Wtime() - time << std::endl;
        time = MPI_Wtime();
        vector2 = greckham_seq(vector);
        std::cout << "Grackham sequential:" << MPI_Wtime() - time << std::endl;
        EXPECT_EQ(vector1.size(), vector2.size());
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
