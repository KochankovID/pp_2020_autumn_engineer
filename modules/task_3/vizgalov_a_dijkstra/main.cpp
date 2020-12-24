// Copyright 2020 Vizgalov Anton
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>

#include <vector>

#include "./dijkstra.h"


TEST(Dijkstra, Dijkstra_Generate_Random) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        ASSERT_NO_THROW(generateRandomGraph(100));
    }
}

TEST(Dijkstra, Dijkstra_Sequential_100) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::vector<int> graph = generateRandomGraph(100);

        double t0 = MPI_Wtime();
        ASSERT_NO_THROW(dijkstraSequential(graph, 0));
        double t1 = MPI_Wtime();

        std::cout << "\nSequential time (graph size 100): " << (t1 - t0) << "\n" << std::endl;
    }
}

TEST(Dijkstra, Dijkstra_Sequential_1000) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        std::vector<int> graph = generateRandomGraph(1000);

        double t0 = MPI_Wtime();
        ASSERT_NO_THROW(dijkstraSequential(graph, 0));
        double t1 = MPI_Wtime();

        std::cout << "\nSequential time (graph size 1000): " << (t1 - t0) << "\n" << std::endl;
    }
}

TEST(Dijkstra, Dijkstra_Parallel_100) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size < 2 || size > 100) return;

    std::vector<int> graph = generateRandomGraph(100);

    double t0 = MPI_Wtime();
    ASSERT_NO_THROW(dijkstraParallel(graph, 0));
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\nParallel time (graph size 100): " << (t1 - t0) << "\n" << std::endl;
    }
}

TEST(Dijkstra, Dijkstra_Parallel_1000) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (size < 2 || size > 1000) return;

    std::vector<int> graph = generateRandomGraph(1000);

    double t0 = MPI_Wtime();
    ASSERT_NO_THROW(dijkstraParallel(graph, 0));
    double t1 = MPI_Wtime();

    if (rank == 0) {
        std::cout << "\nParallel time (graph size 1000): " << (t1 - t0) << "\n" << std::endl;
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
