// Copyright 2020 Stoicheva Darya
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include "sort_batcher_mpi.hpp"


template<typename T>
std::vector<T> create_random_vector(size_t size) {
    assert(size >= 0);
    std::vector<T> vector(size);
    std::mt19937 gen(uint32_t(time(0)));
    std::uniform_int_distribution<int> uid_int(-50, +50);
    std::uniform_real_distribution<float> uid_real(-50.0, +50.0);

    for (size_t i = 0; i < size; i++) {
        vector[i] = static_cast<T>(std::is_same<T, int>::value ? uid_int(gen) : uid_real(gen));
    }

    return vector;
}


template<typename T>
void test_sort_with(std::vector<T> source) {
    std::vector<T> expected = getSequentialOperations<T>(source);

    test_sort_with(source, expected);
}


template<typename T>
void test_sort_with(std::vector<T> source, std::vector<T> expected) {
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::vector<T> result = getParallelOperations<T>(source);

    if (rank == 0) {
        ASSERT_EQ(source.size(), result.size()) << "Vectors source and result are of unequal length";
        ASSERT_EQ(source.size(), expected.size()) << "Vectors source and expected are of unequal length";

        ASSERT_EQ(expected, result);
    }
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_EmptyArray) {
    std::vector<int> source({});
    std::vector<int> expected({});

    test_sort_with<int>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_OneElement) {
    std::vector<int> source({777});
    std::vector<int> expected({777});

    test_sort_with<int>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_TenElements) {
    std::vector<int> source({ 5, 2, 1, 6, 0, 8, 4, 3, 9, 7 });
    std::vector<int> expected({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 });

    test_sort_with<int>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_10OfIntElements) {
    std::vector<int> source = create_random_vector<int>(10);
    std::vector<int> expected(source.begin(), source.end());
    qsort(expected.data(), expected.size(), sizeof(int), compare<int>);

    test_sort_with<int>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_TwentyElements) {
    std::vector<int> source({ 5, 2, 1, 6, 0, 8, 4, 3, 9, 7, 15, 12, 11, 16, 10, 18, 14, 13, 19, 17});
    std::vector<int> expected({ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19 });

    test_sort_with<int>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_TwentyElementsThatFails) {
    std::vector<int> source({-39, 4, -13, 20, -48, 47, 32, 40, -13, -19, 38, -45, -4, -28, 43, -40,
                            45, 15, -13, -27});
    std::vector<int> expected({-48, -45, -40, -39, -28, -27, -19, -13, -13, -13, -4, 4, 15, 20, 32,
                            38, 40, 43, 45, 47});

    test_sort_with<int>(source, expected);
}


TEST(Operations_MPI, Test_sort_Batcher_32ElementsThatFails) {
    std::vector<int> source({31, 9, 31, 47, -23, -13, 44, 29, -5, -2, 21, 12, -26, -49, 46, -30, -4, -33,
                            -19, -7, 10, -26, 43, -11, 1, 20, -22, 24, -21, -40, 33, 24});
    std::vector<int> expected({-49, -40, -33, -30, -26, -26, -23, -22, -21, -19, -13, -11, -7, -5, -4, -2,
                            1, 9, 10, 12, 20, 21, 24, 24, 29, 31, 31, 33, 43, 44, 46, 47});

    test_sort_with<int>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_20OfIntElements) {
    std::vector<int> source = create_random_vector<int>(20);
    std::vector<int> expected(source.begin(), source.end());
    qsort(expected.data(), expected.size(), sizeof(int), compare<int>);

    test_sort_with<int>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_ALotOfIntElements) {
    std::vector<int> source = create_random_vector<int>(10000);
    std::vector<int> expected(source.begin(), source.end());
    qsort(expected.data(), expected.size(), sizeof(int), compare<int>);

    test_sort_with<int>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_ALotOfFloatElements) {
    std::vector<float> source = create_random_vector<float>(10000);
    std::vector<float> expected(source.begin(), source.end());
    qsort(expected.data(), expected.size(), sizeof(float), compare<float>);

    test_sort_with<float>(source, expected);
}


TEST(Operations_MPI, Test_sort_Batcher_20OfFloatElements) {
    std::vector<float> source = create_random_vector<float>(20);
    std::vector<float> expected(source.begin(), source.end());
    qsort(expected.data(), expected.size(), sizeof(float), compare<float>);

    test_sort_with<float>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_20OfDoubleElements) {
    std::vector<double> source = create_random_vector<double>(10);
    std::vector<double> expected(source.begin(), source.end());
    qsort(expected.data(), expected.size(), sizeof(double), compare<double>);

    test_sort_with<double>(source, expected);
}


TEST(Parallel_Operations_MPI, Test_sort_Batcher_ALotOfDoubleElements) {
    std::vector<double> source = create_random_vector<double>(10000);
    std::vector<double> expected(source.begin(), source.end());

    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double t1, t2;
    if (rank == 0) {
        t1 = MPI_Wtime();
        qsort(expected.data(), expected.size(), sizeof(double), compare<double>);
        t1 = MPI_Wtime() - t1;
        t2 = MPI_Wtime();
    }

    test_sort_with<double>(source, expected);

    if (rank == 0) {
        t2 = MPI_Wtime() - t2;
        std::cout << "MPI Batcher sort time: " << t1 << std::endl;
        std::cout << "qsort time: " << t2 << std::endl;
        std::cout << "qsort " << t1 / t2 << " times faster." << std::endl;
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
