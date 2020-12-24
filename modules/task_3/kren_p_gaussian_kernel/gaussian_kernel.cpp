// Copyright 2020 Kren Polina
#include <mpi.h>
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <numeric>
#include "../../../modules/task_3/kren_p_gaussian_kernel/gaussian_kernel.h"


std::vector <int> get_Img(int rows, int columns) {
    if (rows <= 0 || columns <= 0) {
        throw - 1;
    }
    std::vector <int> matrix(rows * columns);
    std::default_random_engine rand_r;
    rand_r.seed(static_cast<unsigned int>(std::time(0)));
    for (int i = 0; i < rows * columns; i++) {
        matrix[i] = rand_r() % 256;
    }
    return matrix;
}


int compute_pixel(const std::vector<int> matrix, const std::vector<int> mask, int i, int j, int rows, int columns) {
    int pixel = 0;
    int mask_size = sqrt(mask.size());
    for (int d = -mask_size / 2; d <= mask_size / 2; ++d) {
        for (int k = -mask_size / 2; k <= mask_size / 2; ++k) {
            int y = j + k;
            int x = i + d;
            if (y < 0 || y > columns - 1) {
                y = j;
            }
            if (x < 0 || x > rows - 1) {
                x = i;
            }
            if (x * columns + y >= columns * rows) {
                x = i;
                y = j;
            }
            pixel += matrix[x * columns + y] * mask[(d + 1) * 3 + k + 1];
        }
    }
    int normal = std::accumulate(mask.begin(), mask.end(), 0);
    if (normal != 0) {
        pixel = std::min(pixel / normal, 255);
    }
    return pixel;
}

std::vector <int> parall_linear_filter(const std::vector <int> mask, std::vector <int> matrix, int rows, int columns) {
    if (static_cast<int>(matrix.size()) != rows * columns) {
        throw - 1;
    }

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int mask_size = sqrt(mask.size());
    int i_ = 0;
    int j_ = 0;
    int rows_ = 0;
    int columns_ = 0;
    int sqrt_size = floor(sqrt(size));
    float sqrt_size_rest = sqrt(size) - sqrt_size;
    int quantity_cols = columns / sqrt_size;
    int rest_cols = columns % sqrt_size;
    int quantity_rows = 0;
    int rest_rows = 0;

    if (sqrt_size_rest == 0) {
        quantity_rows = rows / sqrt_size;
        rest_rows = rows % sqrt_size;
    } else if (fabs(sqrt_size - sqrt_size_rest) < 1) {
        quantity_rows = rows - static_cast<int>((rows / (sqrt_size + 0.5))) * sqrt_size;
        rest_rows = static_cast<int>(rows / (sqrt_size + 0.5)) - quantity_rows;
    } else {
        quantity_rows = rows / (sqrt_size + 1);
        rest_rows = rows % (sqrt_size + 1);
    }
    int rule = 0;
    if ((size - (rows / quantity_rows - 1) * sqrt_size) == 0) {
        rule = 1;
    } else {
        rule = columns / (size - (rows / quantity_rows - 1) * sqrt_size) >= mask_size ? 1 : 0;
    }
    if (quantity_rows >= mask_size && quantity_cols >= mask_size && size > 1 && rule == 1) {
        MPI_Bcast(&matrix[0], rows * columns, MPI_INT, 0, MPI_COMM_WORLD);
        int count = size - (rows / quantity_rows - 1) * sqrt_size;
        if (rank >= (rows / quantity_rows - 1) * sqrt_size && sqrt_size_rest != 0 && size > 2) {
            sqrt_size = size - (rows / quantity_rows - 1) * sqrt_size;
            quantity_cols = columns / count;
            rest_cols = columns % count;
            i_ = (rows / quantity_rows - 1) * quantity_rows;
            j_ = (sqrt_size - (size - rank)) * quantity_cols;
            if ((size - rank) % sqrt_size < rest_cols) {
                columns_ = 1;
            }
            if (j_ > 0 && rest_cols > 0) {
                j_ += (sqrt_size - (size - rank) - 1);
                if ((sqrt_size - (size - rank) - 1) == 0) {
                    j_ += rest_cols;
                }
            }
            rows_ = quantity_rows + rest_rows;
            columns_ += quantity_cols;
        } else {
            i_ = (rank / (sqrt_size)) * quantity_rows;
            if (rank % sqrt_size < rest_cols) {
                j_ = rank % sqrt_size;
                columns_ = 1;
            } else {
                j_ = rest_cols;
            }
            j_ += (rank % sqrt_size) * quantity_cols;
            if (rank >= size - count) {
                rows_ = rest_rows;
            }
            rows_ += quantity_rows;
            columns_ += quantity_cols;
        }
    } else {
        i_ = 0;
        j_ = 0;
        rows_ = rows;
        columns_ = columns;
    }

    std::vector<int> result(rows * columns);

    if (rank == 0) {
        for (int i = i_; i < i_ + rows_; i++) {
            for (int j = j_; j < j_ + columns_; j++) {
                result[i * columns + j] = compute_pixel(matrix, mask, i, j, rows, columns);
            }
        }
        if (quantity_rows >= mask_size && quantity_cols >= mask_size && size > 1 && rule == 1) {
            int proc = 0;
            int amount_cols_ = quantity_cols, rest_cols_ = rest_cols;
            MPI_Status status;
            int sqrt_size_ = sqrt_size;
            int i1 = 0, l_ = 0;
            int flag_ = 0, flag = 1;
            if (sqrt_size == 1) {
                i1++;
                flag = 0;
                flag_ = 1;
                l_++;
            }
            for (int i = i1; i < rows / quantity_rows; i++) {
                for (int j = 0; j < quantity_rows; j++) {
                    if (proc * sqrt_size + flag_ >= (rows / quantity_rows - 1) * sqrt_size
                        && sqrt_size_rest != 0 && size > 2) {
                        amount_cols_ = columns / (size - (rows / quantity_rows - 1) * sqrt_size);
                        rest_cols_ = columns % (size - (rows / quantity_rows - 1) * sqrt_size);
                        sqrt_size_ = size - (rows / quantity_rows - 1) * sqrt_size;
                    }
                    int l1 = 0;
                    if (flag == 1) {
                        l1++;
                    }
                    for (int l = l1; l < sqrt_size_; l++) {
                        int rest = 0, c = 0;
                        if (l >= rest_cols_) {
                            rest = rest_cols_;
                        } else {
                            rest = l;
                            c = 1;
                        }
                        MPI_Recv(&result[(i  * quantity_rows + j) * columns + l * amount_cols_ + rest],
                            amount_cols_ + c, MPI_INT, proc * sqrt_size + l + l_, 1, MPI_COMM_WORLD, &status);
                    }
                }
                flag = 0;
                amount_cols_ = quantity_cols;
                rest_cols_ = rest_cols;
                sqrt_size_ = sqrt_size;
                proc++;
                if (proc == size - 1) {
                    break;
                }
            }
            amount_cols_ = quantity_cols;
            rest_cols_ = rest_cols;
            for (int i = 0; i < rest_rows; i++) {
                for (int l = 0; l < size - (rows / quantity_rows - 1) * sqrt_size; l++) {
                    if (proc * sqrt_size + l >= (rows / quantity_rows - 1)* sqrt_size &&
                        sqrt_size_rest != 0 && size > 2) {
                        amount_cols_ = columns / (size - (rows / quantity_rows - 1) * sqrt_size);
                        rest_cols_ = columns % (size - (rows / quantity_rows - 1) * sqrt_size);
                    }
                    int rest = 0, c = 0;;
                    if (l >= rest_cols_) {
                        rest = rest_cols_;
                    } else {
                        c = 1;
                        rest = l;
                    }
                    MPI_Recv(&result[(rows - rest_rows + i) * columns + l * amount_cols_ + rest],
                        amount_cols_ + c, MPI_INT, (rows / quantity_rows - 1) * sqrt_size + l,
                        1, MPI_COMM_WORLD, &status);
                }
            }
        }
    } else {
        std::vector<int> result_;
        if (quantity_rows >= mask_size && quantity_cols >= mask_size && rule == 1) {
            for (int i = i_; i < i_ + rows_; i++) {
                for (int j = j_; j < j_ + columns_; j++) {
                    result_.push_back(compute_pixel(matrix, mask, i, j, rows, columns));
                }
                MPI_Send(&result_[0], columns_, MPI_INT, 0, 1, MPI_COMM_WORLD);
                result_.clear();
            }
        }
    }
    return result;
}

std::vector <int> SequentialLinearFilter(const std::vector <int> mask,
    std::vector <int> matrix, int rows, int columns) {
    if (static_cast<int>(matrix.size()) != rows * columns) {
        throw - 1;
    }
    std::vector <int> result;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < columns; j++) {
            result.push_back(compute_pixel(matrix, mask, i, j, rows, columns));
        }
    }
    return result;
}
