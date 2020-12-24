// Copyright 2020 Konstantin Sandalov

#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "../../modules/task_3/sandalov_k_histogram_stretching/histogram_stretching.h"

void myCstmTest(size_t height, size_t width);

TEST(Parallel_Operations_MPI, ZeroImage) {
    myCstmTest(0, 0);
}

TEST(Parallel_Operations_MPI, SmallImage) {
    myCstmTest(10, 10);
}

TEST(Parallel_Operations_MPI, MediumImage) {
    myCstmTest(100, 100);
}

TEST(Parallel_Operations_MPI, BigImage) {
    myCstmTest(1000, 1000);
}

void myCstmTest(size_t height, size_t width) {
    int procRank;
    int procSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);
    MPI_Comm_size(MPI_COMM_WORLD, &procSize);
    std::vector<int> randImg;
    std::vector<int> copyImg;
    if (procRank == 0) {
        createRandomImage(&randImg, height, width);
        copyImg = randImg;
    }
    int parStrFin = parallelHistogramStretching(&randImg);
    if (procRank == 0) {
        int seqStrFin = seqHistogramStretching(&copyImg);
        ASSERT_EQ(randImg, copyImg);
        if (height > 0 && width > 0)
            ASSERT_EQ(seqStrFin, 1);
        else
            ASSERT_EQ(seqStrFin, -2);
    }
    if (procSize == 1 && (height == 0 || width == 0))
        ASSERT_EQ(parStrFin, -2);
    else
        ASSERT_EQ(parStrFin, 1);
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
