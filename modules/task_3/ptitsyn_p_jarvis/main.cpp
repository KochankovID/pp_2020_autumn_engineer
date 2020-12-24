// Copyright 2018 Nesterov Alexander
#include <mpi.h>
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./jarvis.h"

TEST(Jarvis_MPI, vector_Size_4) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int count_size_string = 4;
    std::vector<Point> points_1(count_size_string);
    std::vector<Point> points_2(count_size_string);
    std::vector<int> p1_x;
    std::vector<int> p1_y;
    std::vector<int> p2_x;
    std::vector<int> p2_y;

    if (rank == 0) {
        points_1[0].x = 4; points_1[0].y = 1;
        points_1[1].x = 1; points_1[1].y = 9;
        points_1[2].x = 7; points_1[2].y = 1;
        points_1[3].x = 10; points_1[3].y = 10;
        points_2 = points_1;
    }

    double startT;
    if (rank == 0)
        startT = MPI_Wtime();
    points_1 = convexHullPar(points_1); {
        if (rank == 0) {
            double endT = MPI_Wtime();
            std::cout << "MPI time: " << endT - startT << std::endl;
        }
    }

    if (rank == 0) {
        for (size_t i = 0; i < points_1.size(); i++) {
            p1_x.push_back(points_1[i].x);
            p1_y.push_back(points_1[i].y);
        }
        double startT = MPI_Wtime();
        points_2 = convexHull(points_2);
        double endT = MPI_Wtime();
        for (size_t i = 0; i < points_2.size(); i++) {
            p2_x.push_back(points_2[i].x);
            p2_y.push_back(points_2[i].y);
        }
        std::cout << "Consistent time: " << endT - startT << std::endl;
        ASSERT_EQ(p1_x, p2_x);
        ASSERT_EQ(p1_y, p2_y);
    }
}

TEST(Jarvis_MPI, r_vector_Size_10) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int count_size_string = 10;
    std::vector<Point> points_1(count_size_string);
    std::vector<Point> points_2(count_size_string);
    std::vector<int> p1_x;
    std::vector<int> p1_y;
    std::vector<int> p2_x;
    std::vector<int> p2_y;

    if (rank == 0) {
        points_1 = createRandomPoints(count_size_string);
        points_2 = points_1;
    }

    points_1 = convexHullPar(points_1);

    if (rank == 0) {
        for (size_t i = 0; i < points_1.size(); i++) {
            p1_x.push_back(points_1[i].x);
            p1_y.push_back(points_1[i].y);
        }
        double startT = MPI_Wtime();
        points_2 = convexHull(points_2);
        double endT = MPI_Wtime();
        for (size_t i = 0; i < points_1.size(); i++) {
            p2_x.push_back(points_1[i].x);
            p2_y.push_back(points_1[i].y);
        }
        std::cout << "Consistent time: " << endT - startT << std::endl;
        ASSERT_EQ(p1_x, p2_x);
        ASSERT_EQ(p1_y, p2_y);
    }
}

TEST(Jarvis_MPI, r_vector_Size_100) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int count_size_string = 100;
    std::vector<Point> points_1(count_size_string);
    std::vector<Point> points_2(count_size_string);
    std::vector<int> p1_x;
    std::vector<int> p1_y;
    std::vector<int> p2_x;
    std::vector<int> p2_y;

    if (rank == 0) {
        points_1 = createRandomPoints(count_size_string);
        points_2 = points_1;
    }
    points_1 = convexHullPar(points_1);
    if (rank == 0) {
        for (size_t i = 0; i < points_1.size(); i++) {
            p1_x.push_back(points_1[i].x);
            p1_y.push_back(points_1[i].y);
        }
        double startT = MPI_Wtime();
        points_2 = convexHull(points_2);
        double endT = MPI_Wtime();
        for (size_t i = 0; i < points_1.size(); i++) {
            p2_x.push_back(points_1[i].x);
            p2_y.push_back(points_1[i].y);
        }
        std::cout << "Consistent time: " << endT - startT << std::endl;
        ASSERT_EQ(p1_x, p2_x);
        ASSERT_EQ(p1_y, p2_y);
    }
}

TEST(Jarvis_MPI, r_vector_Size_1000) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int count_size_string = 1000;
    std::vector<Point> points_1(count_size_string);
    std::vector<Point> points_2(count_size_string);
    std::vector<int> p1_x;
    std::vector<int> p1_y;
    std::vector<int> p2_x;
    std::vector<int> p2_y;

    if (rank == 0) {
        points_1 = createRandomPoints(count_size_string);
        points_2 = points_1;
    }
    points_1 = convexHullPar(points_1); {
    }

    if (rank == 0) {
        for (size_t i = 0; i < points_1.size(); i++) {
            p1_x.push_back(points_1[i].x);
            p1_y.push_back(points_1[i].y);
        }
        double startT = MPI_Wtime();
        points_2 = convexHull(points_2);
        double endT = MPI_Wtime();
        for (size_t i = 0; i < points_1.size(); i++) {
            p2_x.push_back(points_1[i].x);
            p2_y.push_back(points_1[i].y);
        }
        std::cout << "Consistent time: " << endT - startT << std::endl;
        ASSERT_EQ(p1_x, p2_x);
        ASSERT_EQ(p1_y, p2_y);
    }
}



TEST(Jarvis_MPI, r_vector_Size_1500) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    const int count_size_string = 1500;
    std::vector<Point> points_1(count_size_string);
    std::vector<Point> points_2(count_size_string);
    std::vector<int> p1_x;
    std::vector<int> p1_y;
    std::vector<int> p2_x;
    std::vector<int> p2_y;

    if (rank == 0) {
        points_1 = createRandomPoints(count_size_string);
        points_2 = points_1;
    }

    points_1 = convexHullPar(points_1);

    if (rank == 0) {
        for (size_t i = 0; i < points_1.size(); i++) {
            p1_x.push_back(points_1[i].x);
            p1_y.push_back(points_1[i].y);
        }
        double startT = MPI_Wtime();
        points_2 = convexHull(points_2);
        double endT = MPI_Wtime();
        for (size_t i = 0; i < points_1.size(); i++) {
            p2_x.push_back(points_1[i].x);
            p2_y.push_back(points_1[i].y);
        }
        std::cout << "Consistent time: " << endT - startT << std::endl;
        ASSERT_EQ(p1_x, p2_x);
        ASSERT_EQ(p1_y, p2_y);
    }
}




int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    MPI_Init(&argc, &argv);

    ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();

    listeners.Release(listeners.default_result_printer());
    listeners.Release(listeners.default_xml_generator());

    listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
    return RUN_ALL_TESTS();
}
