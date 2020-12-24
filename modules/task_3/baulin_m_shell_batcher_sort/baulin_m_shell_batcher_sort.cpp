// Copyright 2020 Mikhail Baulin
#include "../../modules/task_3/baulin_m_shell_batcher_sort/baulin_m_shell_batcher_sort.h"
#include <mpi.h>
#include <random>
#include <vector>
#include <climits>
#include <ctime>
#include <algorithm>
#include <utility>

std::vector<std::pair<int, int>> comps;

void S(std::vector<int> procs_up, std::vector<int> procs_down) {
    int proc_count = procs_up.size() + procs_down.size();
    if (proc_count == 1) {
        return;
    } else if (proc_count == 2) {
        comps.push_back(std::pair<int, int>{ procs_up[0], procs_down[0] });
        return;
    }

    std::vector<int> procs_up_odd;
    std::vector<int> procs_up_even;
    std::vector<int> procs_down_odd;
    std::vector<int> procs_down_even;

    for (size_t i = 0; i < procs_up.size(); ++i) {
        if (i % 2) {
            procs_up_even.push_back(procs_up[i]);
        } else {
            procs_up_odd.push_back(procs_up[i]);
        }
    }

    for (size_t i = 0; i < procs_down.size(); ++i) {
        if (i % 2) {
            procs_down_even.push_back(procs_down[i]);
        } else {
            procs_down_odd.push_back(procs_down[i]);
        }
    }

    S(procs_up_odd, procs_down_odd);
    S(procs_up_even, procs_down_even);

    std::vector<int> result(proc_count);

    std::merge(procs_up.begin(), procs_up.end(),
        procs_down.begin(), procs_down.end(),
        result.begin());

    for (size_t i = 1; i + 1 < procs_up.size() + procs_down.size(); i += 2) {
        comps.push_back(std::pair<int, int>{ result[i], result[i + 1]});
    }
}

void B(std::vector<int> procs) {
    if (procs.size() == 1) {
        return;
    }

    std::vector<int> procs_up = std::vector<int>(procs.size() / 2);
    std::vector<int> procs_down = std::vector<int>(procs.size() - procs_up.size());

    std::copy(procs.begin(), procs.begin() + procs_up.size(), procs_up.begin());
    std::copy(procs.begin() + procs_up.size(), procs.end(), procs_down.begin());

    B(procs_up);
    B(procs_down);
    S(procs_up, procs_down);
}

std::vector<int> getVectorWithRandomNumbers(int size, int limit) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::uniform_int_distribution<> dist(0, limit);
    std::vector<int> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = (dist(gen));
    }

    return vec;
}

void getSequentialOperations(std::vector<int>* vec) {
    for (size_t step = (*vec).size() / 2; step > 0; step /= 2)
        for (size_t i = step; i < (*vec).size(); i++)
            for (int j = i - step; j >= 0 && (*vec)[j] > (*vec)[j + step]; j -= step) {
                int tmp = (*vec)[j];
                (*vec)[j] = (*vec)[j + step];
                (*vec)[j + step] = tmp;
            }
}

std::vector<int> getParallelOperations(std::vector<int> vec) {
    int size, rank;

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;

    const int originalSize = vec.size();

    if (rank == 0) {
        while (vec.size() % size != 0) {
            vec.push_back(INT_MAX);
        }
    }

    int delta = vec.size() / size;
    MPI_Bcast(&delta, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int> local_vec(delta);
    if (rank == 0) {
        for (int proc = 1; proc < size; proc++) {
            MPI_Send(&vec[proc * delta], delta,
                MPI_INT, proc, 0, MPI_COMM_WORLD);
        }
        local_vec = vec;
        local_vec.resize(delta);
    } else {
        MPI_Recv(&local_vec[0], delta, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    std::vector<int> global_result(vec.size());;
    getSequentialOperations(&local_vec);

    std::vector<int> procs(size);
    for (int i = 0; i < size; i++) {
        procs[i] = i;
    }
    B(procs);


    std::vector<int> comparative_vec(local_vec.size());
    std::vector<int> temp_vec(local_vec.size());
    for (auto comparator : comps) {
        if (rank == comparator.first) {
            MPI_Send(&local_vec[0], delta, MPI_INT, comparator.second, 0, MPI_COMM_WORLD);
            MPI_Recv(&comparative_vec[0], delta, MPI_INT, comparator.second, 0, MPI_COMM_WORLD, &status);

            for (int res_index = 0, cur_index = 0, tmp_index = 0; tmp_index < delta; tmp_index++) {
                int result = local_vec[res_index];
                int current = comparative_vec[cur_index];
                if (result < current) {
                    temp_vec[tmp_index] = result;
                    res_index++;
                } else {
                    temp_vec[tmp_index] = current;
                    cur_index++;
                }
            }

            local_vec = temp_vec;
        } else if (rank == comparator.second) {
            MPI_Recv(&comparative_vec[0], delta, MPI_INT, comparator.first, 0, MPI_COMM_WORLD, &status);
            MPI_Send(&local_vec[0], delta, MPI_INT, comparator.first, 0, MPI_COMM_WORLD);

            int start = delta - 1;
            for (int res_index = start, cur_index = start, tmp_index = start; tmp_index >= 0; tmp_index--) {
                int result = local_vec[res_index];
                int current = comparative_vec[cur_index];
                if (result > current) {
                    temp_vec[tmp_index] = result;
                    res_index--;
                } else {
                    temp_vec[tmp_index] = current;
                    cur_index--;
                }
            }

            local_vec = temp_vec;
        }
    }

    MPI_Gather(&local_vec[0], delta, MPI_INT, &global_result[0], delta, MPI_INT, 0, MPI_COMM_WORLD);
    global_result.resize(originalSize);
    return global_result;
}
