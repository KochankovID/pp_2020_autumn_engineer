// Copyright 2020 Tronin Dmitry
#include <mpi.h>
#include <random>
#include <vector>
#include "../../../modules/task_3/tronin_d_fox_matrix_multiplication/fox_matrix_multiplication.h"

std::vector<double> getRandomVector(size_t size) {
    std::random_device random_device;
    std::mt19937 rand_gen(random_device());
    std::uniform_real_distribution<double> dist(-1000, 1000);
    std::vector<double> vector(size * size);

    for (size_t i = 0; i < size; ++i) {
        for (size_t j = 0; j < size; ++j) {
            vector[i * size + j] = dist(rand_gen);
        }
    }
    return vector;
}

std::vector<double> SequentialMultiplication(const std::vector<double>& lhs,
                              const std::vector<double>& rhs,
                              size_t n) {
    std::vector<double> result_matrix(n * n, 0);
    for (size_t row = 0; row < n; ++row) {
        for (size_t column = 0; column < n; ++column) {
            for (size_t index = 0; index < n; ++index) {
                result_matrix[row * n + column] +=
                    lhs[row * n + index] * rhs[index * n + column];
            }
        }
    }
    return result_matrix;
}

std::vector<double> ParallelFoxMultiplication(const std::vector<double>& lhs,
                                              const std::vector<double>& rhs,
                                              size_t n) {
    int comm_size, comm_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);

    size_t block_size = n / sqrt(comm_size);
    size_t grid_size = sqrt(comm_size);

    std::vector<int>
        ndim{static_cast<int>(grid_size), static_cast<int>(grid_size)};
    std::vector<int> periods{true, true};

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD,
                    2,
                    ndim.data(),
                    periods.data(),
                    true,
                    &cart_comm);

    MPI_Datatype type, submatrix_type;

    std::vector<int> sizes{static_cast<int>(n), static_cast<int>(n)};

    std::vector<int>
        subsizes{static_cast<int>(block_size), static_cast<int>(block_size)};
    std::vector<int> starts{0, 0};

    MPI_Type_create_subarray(2,
                             sizes.data(),
                             subsizes.data(),
                             starts.data(),
                             MPI_ORDER_C,
                             MPI_DOUBLE,
                             &type);

    MPI_Type_create_resized(type,
                            0,
                            block_size * sizeof(double),
                            &submatrix_type);

    MPI_Type_commit(&submatrix_type);

    std::vector<int> send_counts(comm_size, 1);
    std::vector<int> places(comm_size);
    if (comm_rank == 0) {
        int current_place = 0;
        for (size_t row = 0; row < grid_size; ++row) {
            current_place = block_size * grid_size * row;
            for (size_t column = 0; column < grid_size; ++column) {
                places[row * grid_size + column] = current_place;
                ++current_place;
            }
        }
    }

    std::vector<double> left_matrix_local(block_size * block_size),
        right_matrix_local(block_size * block_size),
        block_result(block_size * block_size, 0);

    MPI_Scatterv(lhs.data(),
                 send_counts.data(),
                 places.data(),
                 submatrix_type,
                 left_matrix_local.data(),
                 block_size * block_size,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Scatterv(rhs.data(),
                 send_counts.data(),
                 places.data(),
                 submatrix_type,
                 right_matrix_local.data(),
                 block_size * block_size,
                 MPI_DOUBLE,
                 0,
                 MPI_COMM_WORLD);

    MPI_Comm row_comm;
    std::vector<int> remain_axis{0, 1};
    MPI_Cart_sub(cart_comm, remain_axis.data(), &row_comm);

    for (size_t iteration = 0; iteration < grid_size; ++iteration) {
        int rank_source, rank_destination;

        std::vector<int> coords(2);
        MPI_Cart_coords(cart_comm, comm_rank, 2, coords.data());

        int source_subgrid_comm_rank;
        std::vector<double> tmp(block_size * block_size, 0);
        if (static_cast<size_t>(coords[1])
            == (coords[0] + iteration) % grid_size) {
            MPI_Comm_rank(row_comm, &source_subgrid_comm_rank);
            tmp = left_matrix_local;
        } else {
            std::vector<int> source_coords
                {(coords[0] + static_cast<int>(iteration))
                     % static_cast<int>(grid_size)};
            MPI_Cart_rank(row_comm,
                          source_coords.data(),
                          &source_subgrid_comm_rank);
        }

        MPI_Bcast(tmp.data(),
                  block_size * block_size,
                  MPI_DOUBLE,
                  source_subgrid_comm_rank,
                  row_comm);

        std::vector<double> current_multiply =
            SequentialMultiplication(
            tmp,
            right_matrix_local,
            block_size);

        for (size_t row = 0; row < block_size; ++row) {
            for (size_t column = 0; column < block_size; ++column) {
                block_result[row * block_size + column] +=
                    current_multiply[row * block_size + column];
            }
        }

        MPI_Status status;

        MPI_Cart_shift(cart_comm, 0, -1, &rank_source, &rank_destination);
        MPI_Sendrecv_replace(right_matrix_local.data(),
                             block_size * block_size,
                             MPI_DOUBLE,
                             rank_destination,
                             0,
                             rank_source,
                             0,
                             MPI_COMM_WORLD,
                             &status);
    }

    std::vector<double> result(n * n);

    MPI_Gatherv(block_result.data(),
                block_size * block_size,
                MPI_DOUBLE,
                result.data(),
                send_counts.data(),
                places.data(),
                submatrix_type,
                0,
                MPI_COMM_WORLD);

    return result;
}
