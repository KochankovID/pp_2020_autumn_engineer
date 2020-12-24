// Copyright 2020 Novikov Danil
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <algorithm>
#include <vector>
#include "./novikov_d_jarvis.h"

TEST(Parallel_Operations_MPI, Test_task_N7) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<point> vec, ip1, ip2;
    int N = 7;
    int vec_size = 20;
    double t;

    if (rank == 0) {
        vec = get_random_image(N, vec_size);
        print_image(vec, N);
        t = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ip1 = convex_hull_jarvis_parallel(vec, vec_size);

    if (rank == 0) {
        std::cout << "Time Parallel:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        ip2 = convex_hull_jarvis_sequential(vec);
        std::cout << "Time Sequential:" << MPI_Wtime() - t << std::endl;
        bool flag = false;
        for (size_t i = 0; i < ip2.size(); i++) {
            if (ip2[i] != ip1[i]) {
                flag = true;
            }
            std::cout << ip2[i].x << ", " << ip2[i].y << std::endl;
        }
        ASSERT_EQ(flag, false);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_N10) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<point> vec, ip1, ip2;
    int N = 10;
    int vec_size = 40;
    double t;

    if (rank == 0) {
        vec = get_random_image(N, vec_size);
        print_image(vec, N);
        t = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ip1 = convex_hull_jarvis_parallel(vec, vec_size);

    if (rank == 0) {
        std::cout << "Time Parallel:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        ip2 = convex_hull_jarvis_sequential(vec);
        std::cout << "Time Sequential:" << MPI_Wtime() - t << std::endl;
        bool flag  = false;
        for (size_t i = 0; i < ip2.size(); i++) {
            if (ip2[i] != ip1[i]) {
                flag = true;
            }
            std::cout << ip2[i].x << ", " << ip2[i].y << std::endl;
        }
        ASSERT_EQ(flag, false);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_N500) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<point> vec, ip1, ip2;
    int N = 5000;
    int vec_size = 5000;
    double t;

    if (rank == 0) {
        vec = get_random_image(N, vec_size);
        t = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ip1 = convex_hull_jarvis_parallel(vec, vec_size);

    if (rank == 0) {
        std::cout << "Time Parallel:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        ip2 = convex_hull_jarvis_sequential(vec);
        std::cout << "Time Sequential:" << MPI_Wtime() - t << std::endl;
        ASSERT_EQ(ip2, ip1);
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task_N10000) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<point> vec, ip1, ip2;
    int N = 10000;
    int vec_size = 10000;
    double t;

    if (rank == 0) {
        vec = get_random_image(N, vec_size);
        t = MPI_Wtime();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ip1 = convex_hull_jarvis_parallel(vec, vec_size);

    if (rank == 0) {
        std::cout << "Time Parallel:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        ip2 = convex_hull_jarvis_sequential(vec);
        std::cout << "Time Sequential:" << MPI_Wtime() - t << std::endl;
        ASSERT_EQ(ip2.size(), ip1.size());
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

TEST(Parallel_Operations_MPI, Test_task__N100000) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    std::vector<point> vec, ip1, ip2;
    int N = 100000;
    int vec_size = 100000;
    double t;

    if (rank == 0) {
        t = MPI_Wtime();
        std::cout << "Generate....." << std::endl;
        vec = get_random_image(N, vec_size);
        std::cout << "Time Generate:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    ip1 = convex_hull_jarvis_parallel(vec, vec_size);

    if (rank == 0) {
        std::cout << "Time Parallel:" << MPI_Wtime() - t << std::endl;
        t = MPI_Wtime();
        ip2 = convex_hull_jarvis_sequential(vec);
        std::cout << "Time Sequential:" << MPI_Wtime() - t << std::endl;
        ASSERT_EQ(ip1.size(), ip2.size());
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
