  // Copyright 2020 Zlobin George
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "./moore_algorithm.h"

TEST(Parallel_Operations_MPI, Test_fillGraphWithRandomValues_No_Throw) {
    int count_of_vertex = 10;
    int* graph = new int[count_of_vertex * count_of_vertex];

    ASSERT_NO_THROW(fillGraphWithRandomValues(graph, count_of_vertex));

    delete[] graph;
}

TEST(Parallel_Operations_MPI, Test_getSequentialMooreAlgorithm_No_Throw) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        int count_of_vertex = 10;
        int* graph = new int[count_of_vertex * count_of_vertex];
        fillGraphWithRandomValues(graph, 10);

        int* shortest_ways = new int[count_of_vertex];

        ASSERT_NO_THROW(getSequentialMooreAlgorithm(graph, count_of_vertex, shortest_ways));

        delete[] graph;
        delete[] shortest_ways;
    }
}

TEST(Parallel_Operations_MPI, Test_getParallelMooreAlgorithm_No_Throw) {
    int count_of_vertex = 10;
    int* graph = new int[count_of_vertex * count_of_vertex];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        fillGraphWithRandomValues(graph, count_of_vertex);
    }
    MPI_Bcast(&graph[0], count_of_vertex * count_of_vertex, MPI_INT, 0, MPI_COMM_WORLD);

    int* shortest_ways = new int[count_of_vertex];

    ASSERT_NO_THROW(getParallelMooreAlgorithm(graph, count_of_vertex, shortest_ways));

    delete[] graph;
    delete[] shortest_ways;
}

TEST(Parallel_Operations_MPI, Test_getSequentialMooreAlgorithm_Correct) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        int count_of_vertex = 3;
        int graph[] = {0, 2, 1000,
                    2, 0, 1,
                    1, 1000, 0};

        int* shortest_ways = new int[count_of_vertex];
        getSequentialMooreAlgorithm(graph, count_of_vertex, shortest_ways, 0);
        int ref_shortest_ways[] = {0, 2, 3};
        for (int i = 0; i < count_of_vertex; i++) {
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways[i]);
        }
        delete[] shortest_ways;
    }
}

TEST(Parallel_Operations_MPI, Test_getParallelMooreAlgorithm_Correct) {
    int count_of_vertex = 3;
    int graph[] = {0, 1000, 8,
                  3, 0, 1000,
                  1, 5, 0};

    int* shortest_ways = new int[count_of_vertex];
    getParallelMooreAlgorithm(graph, count_of_vertex, shortest_ways, 0);
    int ref_shortest_ways[] = {0, 13, 8};

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        for (int i = 0; i < count_of_vertex; i++) {
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways[i]);
        }
    }

    delete[] shortest_ways;
}

TEST(Parallel_Operations_MPI, Test_Parallel_And_Sequential_Has_Same_Result_StaticCase1) {
    int count_of_vertex = 3;
    int graph[] = {0, 1000, 22,
                  1000, 0, 1000,
                  1000, 1000, 0};

    int* shortest_ways_seq = new int[count_of_vertex];
    int* shortest_ways_par = new int[count_of_vertex];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        getSequentialMooreAlgorithm(graph, count_of_vertex, shortest_ways_seq, 0);
    }
    getParallelMooreAlgorithm(graph, count_of_vertex, shortest_ways_par, 0);
    int ref_shortest_ways[] = {0, 1000, 22};

    if (rank == 0) {
        for (int i = 0; i < count_of_vertex; i++) {
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways_seq[i]);
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways_par[i]);
        }
    }

    delete[] shortest_ways_seq;
    delete[] shortest_ways_par;
}

TEST(Parallel_Operations_MPI, Test_Parallel_And_Sequential_Has_Same_Result_StaticCase2) {
    int count_of_vertex = 4;
    int graph[] = {0, 1000, 22, 1000,
                  1000, 0, 1000, 4,
                  1000, 2, 0, 20,
                  10, 20, 40, 0};

    int* shortest_ways_seq = new int[count_of_vertex];
    int* shortest_ways_par = new int[count_of_vertex];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        getSequentialMooreAlgorithm(graph, count_of_vertex, shortest_ways_seq, 0);
    }
    getParallelMooreAlgorithm(graph, count_of_vertex, shortest_ways_par, 0);
    int ref_shortest_ways[] = {0, 24, 22, 28};

    if (rank == 0) {
        for (int i = 0; i < count_of_vertex; i++) {
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways_seq[i]);
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways_par[i]);
        }
    }

    delete[] shortest_ways_seq;
    delete[] shortest_ways_par;
}

TEST(Parallel_Operations_MPI, Test_Parallel_And_Sequential_Has_Same_Result_StaticCase3) {
    int count_of_vertex = 4;
    int graph[] = {0, 33, 22, 44,
                  1000, 0, 1000, 1000,
                  1, 2, 0, 3,
                  1000, 1000, 1000, 0};

    int* shortest_ways_seq = new int[count_of_vertex];
    int* shortest_ways_par = new int[count_of_vertex];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        getSequentialMooreAlgorithm(graph, count_of_vertex, shortest_ways_seq, 0);
    }
    getParallelMooreAlgorithm(graph, count_of_vertex, shortest_ways_par, 0);
    int ref_shortest_ways[] = {0, 24, 22, 25};

    if (rank == 0) {
        for (int i = 0; i < count_of_vertex; i++) {
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways_seq[i]);
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways_par[i]);
        }
    }
    delete[] shortest_ways_seq;
    delete[] shortest_ways_par;
}

TEST(Parallel_Operations_MPI, Test_Parallel_And_Sequential_Has_Same_Result_StaticCase4) {
    int count_of_vertex = 5;
    int graph[] = {0, 2, 1000, 1000, 1000,
                  1000, 0, 3, 1000, 1000,
                  1, 1000, 0, 1000, 1000,
                  1000, 1000, 1000, 0, 5,
                  1000, 1000, 1000, 4, 0};

    int* shortest_ways_seq = new int[count_of_vertex];
    int* shortest_ways_par = new int[count_of_vertex];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        getSequentialMooreAlgorithm(graph, count_of_vertex, shortest_ways_seq, 0);
    }
    getParallelMooreAlgorithm(graph, count_of_vertex, shortest_ways_par, 0);
    int ref_shortest_ways[] = {0, 2, 5, 1000, 1000};

    if (rank == 0) {
        for (int i = 0; i < count_of_vertex; i++) {
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways_seq[i]);
            ASSERT_EQ(ref_shortest_ways[i], shortest_ways_par[i]);
        }
    }
    delete[] shortest_ways_seq;
    delete[] shortest_ways_par;
}

TEST(Parallel_Operations_MPI, Test_Parallel_And_Sequential_Has_Same_Result_RandomCase) {
    int count_of_vertex = 100;
    int* graph = new int[count_of_vertex * count_of_vertex];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        fillGraphWithRandomValues(graph, count_of_vertex);
    }
    MPI_Bcast(&graph[0], count_of_vertex * count_of_vertex, MPI_INT, 0, MPI_COMM_WORLD);

    int* shortest_ways_seq = new int[count_of_vertex];
    int* shortest_ways_par = new int[count_of_vertex];
    if (rank == 0) {
        getSequentialMooreAlgorithm(graph, count_of_vertex, shortest_ways_seq, 0);
    }
    getParallelMooreAlgorithm(graph, count_of_vertex, shortest_ways_par, 0);

    if (rank == 0) {
        for (int i = 0; i < count_of_vertex; i++) {
            ASSERT_EQ(shortest_ways_seq[i], shortest_ways_par[i]);
        }
    }
    delete[] shortest_ways_seq;
    delete[] shortest_ways_par;
    delete[] graph;
}

TEST(Parallel_Operations_MPI, DISABLED_Test_Compare_Performance_Parallel_And_Sequential) {
    int count_of_vertex = 20000;
    int* graph = new int[count_of_vertex * count_of_vertex];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        fillGraphWithRandomValues(graph, count_of_vertex);
    }
    MPI_Bcast(&graph[0], count_of_vertex * count_of_vertex, MPI_INT, 0, MPI_COMM_WORLD);

    int* shortest_ways_seq = new int[count_of_vertex];
    int* shortest_ways_par = new int[count_of_vertex];

    double start, time_seq, time_par;
    start = MPI_Wtime();
    if (rank == 0) {
        getSequentialMooreAlgorithm(graph, count_of_vertex, shortest_ways_seq, 0);
    }
    time_seq = MPI_Wtime() - start;
    start = MPI_Wtime();
    getParallelMooreAlgorithm(graph, count_of_vertex, shortest_ways_par, 0);
    time_par = MPI_Wtime() - start;

    if (rank == 0) {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        if (size > 1) {
            std::cout << "[ INFO     ] Num processes: " << size <<"\n";
            std::cout << "[ INFO     ] Time parallel moore: " << time_par <<"\n";
            std::cout << "[ INFO     ] Time squential moore: " << time_seq <<"\n";
        }
            ASSERT_TRUE(time_par < time_seq);
        for (int i = 0; i < count_of_vertex; i++) {
            ASSERT_EQ(shortest_ways_seq[i], shortest_ways_par[i]);
        }
    }

    delete[] shortest_ways_seq;
    delete[] shortest_ways_par;
    delete[] graph;
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
