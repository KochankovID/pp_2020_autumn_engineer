// Copyright 2020 Eremina Alena
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <random>
#include <vector>
#include "./ring.h"

TEST(Topology_Ring, Default_Number_Of_Neighboring_Processes_Is_2) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm RingComm;

    /* 0 - 1 - 2 - ... - (ProcNum-1) */
    RingComm = topologyRing(NULL);

    if (ProcRank == 0) {
        for (int proc = 0; proc < ProcNum; proc++) {
            int nneighbors;
            MPI_Graph_neighbors_count(RingComm, proc, &nneighbors);
            ASSERT_EQ(nneighbors, 2);
        }
    }
}

TEST(Topology_Ring, Default_Left_Neighbor_Process_Rank_Is_Correct) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm RingComm;

    /* 0 - 1 - 2 - ... - (ProcNum-1) */
    RingComm = topologyRing(NULL);

    if (ProcRank == 0) {
        for (int rank = 0; rank < ProcNum; rank++) {
            int neighbors[2];
            MPI_Graph_neighbors(RingComm, rank, 2, &neighbors[0]);
            ASSERT_EQ((neighbors[0] + 1) % ProcNum, rank);
        }
    }
}

TEST(Topology_Ring, Default_Right_Neighbor_Process_Rank_Is_Correct) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm RingComm;

    /* 0 - 1 - 2 - ... - (ProcNum-1) */
    RingComm = topologyRing(NULL);

    if (ProcRank == 0) {
        for (int rank = 0; rank < ProcNum; rank++) {
            int neighbors[2];
            MPI_Graph_neighbors(RingComm, rank, 2, &neighbors[0]);
            ASSERT_EQ(neighbors[1], (rank + 1) % ProcNum);
        }
    }
}

TEST(Topology_Ring, Ordered_Number_Of_Neighboring_Processes_Is_2) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm RingComm;

    /* if ProcNum = 10 */
    /* 0 - 2 - 4 - 6 - 8 - 9 - 7 - 5 - 3 - 1 */
    int *order = new int[ProcNum];
    for (int i = 0; i < ProcNum/2; i++) {
        order[i] = i*2;
        order[ProcNum - 1 - i] = i * 2 + 1;
    }
    if (ProcNum % 2 == 1) order[ProcNum / 2] = ProcNum - 1;

    RingComm = topologyRing(order);

    if (ProcRank == 0) {
        for (int proc = 0; proc < ProcNum; proc++) {
            int nneighbors;
            MPI_Graph_neighbors_count(RingComm, proc, &nneighbors);
            ASSERT_EQ(nneighbors, 2);
        }
    }
}

TEST(Topology_Ring, Ordered_Left_Neighbor_Process_Rank_Is_Correct) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm RingComm;

    /* if ProcNum = 10 */
    /* 0 - 2 - 4 - 6 - 8 - 9 - 7 - 5 - 3 - 1 */
    int *order = new int[ProcNum];
    for (int i = 0; i < ProcNum / 2; i++) {
        order[i] = i * 2;
        order[ProcNum - 1 - i] = i * 2 + 1;
    }
    if (ProcNum % 2 == 1) order[ProcNum / 2] = ProcNum - 1;

    RingComm = topologyRing(order);

    if (ProcRank == 0) {
        for (int i = 0; i < ProcNum; i++) {
            int neighbors[2];
            MPI_Graph_neighbors(RingComm, i, 2, &neighbors[0]);
            int left;
            int ptr;
            for (ptr = 0; ptr < ProcNum; ptr++) {
                if (order[ptr] == i) break;
            }
            if (ptr == 0)
                left = order[ProcNum - 1];
            else
                left = order[ptr - 1];
            ASSERT_EQ(neighbors[0], left);
        }
    }
}

TEST(Topology_Ring, Ordered_Right_Neighbor_Process_Rank_Is_Correct) {
    int ProcRank, ProcNum;
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm RingComm;

    /* if ProcNum = 10 */
    /* 0 - 2 - 4 - 6 - 8 - 9 - 7 - 5 - 3 - 1 */
    int *order = new int[ProcNum];
    for (int i = 0; i < ProcNum / 2; i++) {
        order[i] = i * 2;
        order[ProcNum - 1 - i] = i * 2 + 1;
    }
    if (ProcNum % 2 == 1) order[ProcNum / 2] = ProcNum - 1;

    RingComm = topologyRing(order);

    if (ProcRank == 0) {
        for (int i = 0; i < ProcNum; i++) {
            int neighbors[2];
            MPI_Graph_neighbors(RingComm, i, 2, &neighbors[0]);
            int right;
            int ptr;
            for (ptr = 0; ptr < ProcNum; ptr++) {
                if (order[ptr] == i)
                    break;
            }
            if (ptr == ProcNum-1)
                right = order[0];
            else
                right = order[ptr + 1];
            ASSERT_EQ(neighbors[1], right);
        }
    }
}

TEST(Topology_Ring, Test_Default_Topology) {
    int ProcRank, ProcNum;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    std::vector<int> array(ProcNum);
    array = getRandomVector(ProcNum);

    MPI_Comm RingComm = topologyRing(NULL);

    int neighbors[2];
    MPI_Graph_neighbors(RingComm, ProcRank, 2, &neighbors[0]);

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Send(&array[ProcRank], 1, MPI_INT, neighbors[1], ProcRank, RingComm);
    MPI_Barrier(MPI_COMM_WORLD);

    int number;
    MPI_Status status;
    MPI_Recv(&number, 1, MPI_INT, neighbors[0], neighbors[0], RingComm, &status);
    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_EQ(array[neighbors[0]], number);
}

TEST(Topology_Ring, Test_Ordered_Topology) {
    int ProcRank, ProcNum;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);

    std::vector<int> array(ProcNum);
    array = getRandomVector(ProcNum);

    /* if ProcNum = 10 */
    /* 0 - 2 - 4 - 6 - 8 - 9 - 7 - 5 - 3 - 1 */
    int *order = new int[ProcNum];
    for (int i = 0; i < ProcNum / 2; i++) {
        order[i] = i * 2;
        order[ProcNum - 1 - i] = i * 2 + 1;
    }
    if (ProcNum % 2 == 1) order[ProcNum / 2] = ProcNum - 1;

    MPI_Comm RingComm = topologyRing(order);

    int neighbors[2];
    MPI_Graph_neighbors(RingComm, ProcRank, 2, &neighbors[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Send(&array[ProcRank], 1, MPI_INT, neighbors[1], ProcRank, RingComm);
    MPI_Barrier(MPI_COMM_WORLD);

    int number;
    MPI_Status status;
    MPI_Recv(&number, 1, MPI_INT, neighbors[0], neighbors[0], RingComm, &status);
    MPI_Barrier(MPI_COMM_WORLD);

    ASSERT_EQ(array[neighbors[0]], number);
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
