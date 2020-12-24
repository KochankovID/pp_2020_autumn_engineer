  // Copyright 2020 Ivanov Yaroslav
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <mpi.h>
#include <iostream>
#include <vector>
#include "./linefilter_block.h"


TEST(Parallel_Operation_MPI, Test_Can_Create_Image_50x50) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int height = 50;
    int width = 50;
    int pixes = height * width;
    int radius = 1;
    float sigma = 5.0;
    std::vector<int> Arr(pixes);
    std::vector<int> Arr2(pixes);
    std::vector<int> Arr1(pixes);
    std::vector<float> Kernel = createKernel(radius, sigma);
    Arr = CreateRandomImg(height, width);
    Arr2 = ParallelFilter(Arr, height, width, radius, sigma, Kernel);
    if (rank == 0) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Arr1[i * width + j] = LocalFilter(Arr, height, width, radius, sigma, Kernel, i, j);
            }
        }
        ASSERT_EQ(Arr1, Arr2);
    }
}

TEST(Parallel_Operation_MPI, Test_Can_Create_Image_10x10) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int height = 10;
    int width = 10;
    int pixes = height * width;
    int radius = 1;
    float sigma = 7.0;
    std::vector<int> Arr(pixes);
    std::vector<int> Arr2(pixes);
    std::vector<int> Arr1(pixes);
    std::vector<float> Kernel = createKernel(radius, sigma);
    Arr = CreateRandomImg(height, width);
    Arr2 = ParallelFilter(Arr, height, width, radius, sigma, Kernel);
    if (rank == 0) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Arr1[i * width + j] = LocalFilter(Arr, height, width, radius, sigma, Kernel, i, j);
            }
        }
        ASSERT_EQ(Arr1, Arr2);
    }
}

TEST(Parallel_Operation_MPI, Test_Can_Create_Image_100x100) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int height = 100;
    int width = 100;
    int pixes = height * width;
    int radius = 1;
    float sigma = 5.0;
    std::vector<int> Arr(pixes);
    std::vector<int> Arr2(pixes);
    std::vector<int> Arr1(pixes);
    std::vector<float> Kernel = createKernel(radius, sigma);
    Arr = CreateRandomImg(height, width);
    Arr2 = ParallelFilter(Arr, height, width, radius, sigma, Kernel);
    if (rank == 0) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Arr1[i * width + j] = LocalFilter(Arr, height, width, radius, sigma, Kernel, i, j);
            }
        }
        ASSERT_EQ(Arr1, Arr2);
    }
}

TEST(Parallel_Operation_MPI, Test_Can_Create_Image_250x250) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int height = 250;
    int width = 250;
    int pixes = height * width;
    int radius = 1;
    float sigma = 5.0;
    std::vector<int> Arr(pixes);
    std::vector<int> Arr2(pixes);
    std::vector<int> Arr1(pixes);
    std::vector<float> Kernel = createKernel(radius, sigma);
    Arr = CreateRandomImg(height, width);
    Arr2 = ParallelFilter(Arr, height, width, radius, sigma, Kernel);
    if (rank == 0) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Arr1[i * width + j] = LocalFilter(Arr, height, width, radius, sigma, Kernel, i, j);
            }
        }
        ASSERT_EQ(Arr1, Arr2);
    }
}

TEST(Parallel_Operation_MPI, Test_Can_Create_Image_500x500) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    int height = 500;
    int width = 500;
    int pixes = height * width;
    int radius = 1;
    float sigma = 5.0;
    std::vector<int> Arr(pixes);
    std::vector<int> Arr2(pixes);
    std::vector<int> Arr1(pixes);
    std::vector<float> Kernel = createKernel(radius, sigma);
    Arr = CreateRandomImg(height, width);
    Arr2 = ParallelFilter(Arr, height, width, radius, sigma, Kernel);
    if (rank == 0) {
        for (int i = 0; i < height; i++) {
            for (int j = 0; j < width; j++) {
                Arr1[i * width + j] = LocalFilter(Arr, height, width, radius, sigma, Kernel, i, j);
            }
        }
        ASSERT_EQ(Arr1, Arr2);
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
