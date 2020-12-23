// Copyright 2020 Kochankov Ilya

#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./bitwise_sort_for_double_with_simple_merge.h"


using std::vector;

TEST(Parallel_Operations_MPI, getRandomVector_negative_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> vec;
    const int size = -999;
    if (rank == 0) {
        ASSERT_ANY_THROW(getRandomVector(size));
    }
}

TEST(Parallel_Operations_MPI, getRandomVector_works) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::vector<double> vec;
    const int size = 100;
    if (rank == 0) {
        vec = getRandomVector(size);
        ASSERT_EQ(static_cast<int>(vec.size()), 100);
    }
}

TEST(Parallel_Operations_MPI, get_digit_works) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    double number = 12.2;
    if (rank == 0) {
        int digit_1 = get_digit(number, -1);
    int digit_2 = get_digit(number, 0);
    int digit_3 = get_digit(number, 1);

    EXPECT_EQ(digit_1, 2);
    EXPECT_EQ(digit_2, 2);
    EXPECT_EQ(digit_3, 1);

    int digit_4 = get_digit(91234.34234, -5);
    EXPECT_EQ(digit_4, 4);
    }
}

TEST(Parallel_Operations_MPI, bitwise_sort_works) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        vector<double> vect = { 1, 6, 3, 9, 2 };
        vector<double> result = { 1, 2, 3, 6, 9 };
        auto sorted = bitwise_sort(vect, 0);
        EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, bitwise_sort_works_double) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        vector<double> vect = { 0.1, 0.6, 0.3, 0.9, 0.2 };
        vector<double> result = { 0.1, 0.2, 0.3, 0.6, 0.9 };
        auto sorted = bitwise_sort(vect, -1);
        EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, bitwise_sort_works_double_wrong_digit) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        vector<double> vect = { 0.1, 0.6, 0.3, 0.9, 0.2 };
        vector<double> result = { 0.1, 0.6, 0.3, 0.9, 0.2 };
        auto sorted = bitwise_sort(vect, -2);
        EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, get_digit_number_above_zero_works) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        double number = 134234;
        auto koll_digit = get_digit_number_above_zero(number);
        EXPECT_EQ(koll_digit, 6);
    }
}

TEST(Parallel_Operations_MPI, get_digit_number_below_zero_works) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        double number = 91234.342300;
        auto koll_digit = get_digit_number_below_zero(number);
        EXPECT_EQ(koll_digit, 6);
    }
}

TEST(Parallel_Operations_MPI, linear_bitwise_sort_works) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
            vector<double> vect = { 111.123, 26.333, 3.444, 91234.34234, 32.1, 32.2 };
            vector<double> result = { 3.444, 26.333, 32.1, 32.2, 111.123, 91234.34234 };
            auto sorted = linear_bitwise_sort(vect);
            EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, parallel_bitwise_sort_works) {
    int rank;
    vector<double> vect(6);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        vector<double> tmp = { 111.123, 26.333, 3.444, 91234.34234, 32.1, 32.2 };
        vect = tmp;
    }
    auto sorted = parallel_bitwise_sort(vect);
    if (rank == 0) {
        vector<double> result = { 3.444, 26.333, 32.1, 32.2, 111.123, 91234.34234 };
        EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, parallel_bitwise_sort_works_rand_vector) {
    int rank;
    double start, end;
    double seconds;

    vector<double> vect(5000000);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        vect = getRandomVector(5000000);
    }
    start = MPI_Wtime();

    auto sorted = parallel_bitwise_sort(vect);

    end = MPI_Wtime();

    if (rank == 0) {
        seconds = end - start;
        std::cout << "Parallel method: " << seconds << std::endl;

        start = MPI_Wtime();
        vector<double> result = linear_bitwise_sort(vect);
        end = MPI_Wtime();
        seconds = end - start;
        std::cout << "Linear method: " << seconds << std::endl;
        EXPECT_EQ(sorted, result);
    }
}

TEST(Parallel_Operations_MPI, merge_works) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::vector<double> v1{ 2, 2, 4, 6 };
        std::vector<double> v2{ 0, 2, 3, 6 };
        std::vector<double> res{ 0, 2, 2, 2, 3, 4, 6, 6 };
        std::vector<double> v3 = merge(v1, v2);
        for (int i = 0; i < 8; ++i)
            EXPECT_EQ(v3[i], res[i]) << "Arrays check_array and res_array differ at index " << i;
    }
}

TEST(Parallel_Operations_MPI, merge_works_diff_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
        std::vector<double> v1{ 26.333, 111.123 };
        std::vector<double> v2{ 32.1, 32.2, 91234.3 };
        std::vector<double> res{ 26.333, 32.1, 32.2, 111.123, 91234.3 };
        std::vector<double> v3 = merge(v1, v2);
        for (int i = 0; i < static_cast<int>(res.size()); ++i)
            EXPECT_EQ(v3[i], res[i]) << "Arrays check_array and res_array differ at index " << i;
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
