// Copyright 2020 Voronin Aleksey
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include <utility>
#include <algorithm>
#include "../../../modules/task_3/voronin_a_batcher_quick_sort/batcher_quick_sort.h"

std::vector<std::pair<int, int>> comparators;

std::vector<int> getRandomVector(int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<int> vector(size);
    for (int i = 0; i < size; i++) {
        vector[i] = gen() % 100;
    }
    return vector;
}

void quickSort(int *vector, int left, int right) {
    int i = left;
    int j = right;
    int mid = vector[(i + j) / 2];

    while (i <= j) {
        while (vector[i] < mid)
            i++;
        while (vector[j] > mid)
            j--;
        if (i <= j) {
            std::swap(vector[i], vector[j]);
            i++;
            j--;
        }
    }
    if (j > left)
        quickSort(vector, left, j);
    if (i < right)
        quickSort(vector, i, right);
}

std::vector<int> quickSort(std::vector<int> vector) {
    quickSort(vector.data(), 0, vector.size() - 1);
    return vector;
}

void makeNetwork(int size) {
    std::vector<int> vector;
    for (int i = 0; i < size; i++) {
        vector.push_back(i);
    }
    net(vector);
}

void net(std::vector<int> vector) {
    int size = vector.size();
    if (size <= 1) {
        return;
    }
    std::vector<int> l(vector.begin(), vector.begin() + size / 2);
    std::vector<int> r(vector.begin() + size / 2, vector.end());
    net(l);
    net(r);
    oddEvenMerge(l, r);
}

void oddEvenMerge(std::vector<int> left, std::vector<int> right) {
    int size = left.size() + right.size();
    if (size <= 1) {
        return;
    }
    if (size == 2) {
        comparators.push_back(std::pair<int, int>(left[0], right[0]));
        return;
    }
    std::vector<int> left_odd;
    std::vector<int> left_even;
    std::vector<int> right_odd;
    std::vector<int> right_even;
    for (unsigned int i = 0; i < left.size(); i++) {
        if (i % 2) {
            left_even.push_back(left[i]);
        } else {
            left_odd.push_back(left[i]);
        }
    }
    for (unsigned int i = 0; i < right.size(); i++) {
        if (i % 2) {
            right_even.push_back(right[i]);
        } else {
            right_odd.push_back(right[i]);
        }
    }

    oddEvenMerge(left_odd, right_odd);
    oddEvenMerge(left_even, right_even);

    std::vector<int> res;
    for (unsigned int i = 0; i < left.size(); i++) {
        res.push_back(left[i]);
    }
    for (unsigned int i = 0; i < right.size(); i++) {
        res.push_back(right[i]);
    }

    for (int i = 1; i + 1 < size; i += 2) {
        comparators.push_back(std::pair<int, int>(res[i], res[i + 1]));
    }
}

std::vector<int> parallelSort(std::vector<int> vector) {
    int rank, size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if ((unsigned int)size >= vector.size()) {
        if (rank == 0) {
            vector = quickSort(vector);
        }
        return vector;
    }
    makeNetwork(size);
    int extension = 0;
    if (rank == 0) {
        while (true) {
            if (vector.size() % size) {
                extension++;
                vector.push_back(101);
            } else {
                break;
            }
        }
    }

    int extended_size = vector.size();
    MPI_Bcast(&extended_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_size = extended_size / size;
    std::vector<int> local_vector(local_size);
    std::vector<int> nearby_vector(local_size);
    std::vector<int> temp(local_size);
    MPI_Scatter(&vector[0], local_size, MPI_INT, &local_vector[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);
    local_vector = quickSort(local_vector);
    for (unsigned int i = 0; i < comparators.size(); i++) {
        if (rank == comparators[i].first) {
            MPI_Send(&local_vector[0], local_size, MPI_INT, comparators[i].second, 0, MPI_COMM_WORLD);
            MPI_Status status;
            MPI_Recv(&nearby_vector[0], local_size, MPI_INT, comparators[i].second, 0, MPI_COMM_WORLD, &status);
            int local_point = 0;
            int nearby_point = 0;
            for (int j = 0; j < local_size; j++) {
                int local = local_vector[local_point];
                int nearby = nearby_vector[nearby_point];
                if (local < nearby) {
                    temp[j] = local;
                    local_point++;
                } else {
                    temp[j] = nearby;
                    nearby_point++;
                }
            }
            local_vector = temp;
        } else if (rank == comparators[i].second) {
            MPI_Status status;
            MPI_Recv(&nearby_vector[0], local_size, MPI_INT, comparators[i].first, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&local_vector[0], local_size, MPI_INT, comparators[i].first, 0, MPI_COMM_WORLD);
            int start = local_size - 1;
            int local_point = start;
            int nearby_point = start;
            for (int j = start; j >= 0; j--) {
                int local = local_vector[local_point];
                int nearby = nearby_vector[nearby_point];
                if (local > nearby) {
                    temp[j] = local;
                    local_point--;
                } else {
                    temp[j] = nearby;
                    nearby_point--;
                }
            }
            local_vector = temp;
        }
    }
    MPI_Gather(&local_vector[0], local_size, MPI_INT, &vector[0], local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0 && extension > 0)
        vector.erase(vector.begin() + extended_size - extension, vector.begin() + vector.size());
    comparators.clear();
    return vector;
}
