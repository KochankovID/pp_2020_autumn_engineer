#include "./radix_sort_double_simple_merge.h"

#include <mpi.h>

#include <ctime>
#include <iostream>
#include <random>
#include <vector>
#include <list>

using namespace std;

vector<double> getRandomVector(int size) {
    vector<double> vec(size);
    for (int i = 0; i < size; i++) {
        std::mt19937 gen;
        gen.seed(static_cast<unsigned char>(time(nullptr)));
        vec[i] = (gen() / 10000. * (1000. - 10.) + 10.);
    }
    return vec;
}

vector<double> bubbleSort(const std::vector<double>& vec) {
    std::vector<double> sorted;
    for (auto i = vec.begin(); i < vec.end(); ++i) {
        for (auto j = i; j < vec.end(); ++j) {
            if (*i > *j) {
                sorted.push_back(*j);
            }else{
                sorted.push_back(*i);
            }
        }
    }
    return sorted;
}

int numberOfPositiveRadix(int num) {
    int num_of_radix = 0;
    while (num> 0) {
        num /= 10;
        num_of_radix++;
    }
    return num_of_radix;
}

int numberOfNegativeRadix(double num) {
    std::string str_num = to_string(num);
    if (str_num.find('.')) {
        return -(static_cast<int>(str_num.find('.')) - static_cast<int>(str_num.size())) - 1;
    } else {
        return 0;
    }
}

vector<double> merge(const vector<double>& vec_left, const vector<double>& vec_right) {
    std::vector<double> result((vec_left.size() + vec_right.size()));

    int i = 0, j = 0, k = 0;
    while (i < vec_left.size() && j < vec_right.size()) {
        if (vec_left[i] < vec_right[j])
            result[k] = vec_left[i++];
        else
            result[k] = vec_right[j++];
        k++;
    }
    while (i < vec_left.size()) {
        result[k++] = vec_left[i++];
    }
    while (j < vec_right.size()) {
        result[k++] = vec_right[j++];
    }

    return result;
}

int getDigit(double num, int radix) {
    if (radix > 0) {
        double mask = pow(10, radix);
        double tmp = num / mask;
        return static_cast<int>(tmp) % 10;
    }
    return uint64_t(num * pow(10, -radix)) % 10;
}

std::vector<double> radixSort(const std::vector<double>& vect, int rad) {
    vector<double> res;
    vector<list<double>> radix(10);
    for (auto i = radix.begin(); i < radix.end(); i++) {
        (*i) = list<double>();
    }
    for (int i = 0; i < vect.size(); i++) {
        radix[getDigit(vect[i], rad)].push_back(vect[i]);
    }
    for (auto vect : radix) {
        for (auto element : vect) {
            res.push_back(element);
        }
    }
    return res;
}

std::vector<double> linearRadixSort(const std::vector<double>& vect) {
    int radixNegativeZero = 0;
    int maxRadixNegativeZero = numberOfNegativeRadix(vect[0]);
    for (auto element : vect) {
        radixNegativeZero = numberOfNegativeRadix(element);
        if (radixNegativeZero > maxRadixNegativeZero) {
            maxRadixNegativeZero = radixNegativeZero;
        }
    }
    double max = vect[0];
    for (int i = 1; i < vect.size(); i++) {
        if (vect[i] > max) {
            max = vect[i];
        }
    }
    int maxRadixPositiveZero = numberOfPositiveRadix(max);
    vector<double> result(vect);
    for (int i = -maxRadixNegativeZero; i <= maxRadixPositiveZero; i++) {
        result = radixSort(result, i);
    }
    return result;
}

std::vector<double> parallelRadixSort(const std::vector<double>& vect) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if ((size == 1) || (vect.size() < size)) {
        if (rank == 0) {
            return linearRadixSort(vect);
        }
        return vector<double>();
    }

    vector<double> local_vect;

    int num_of_local_numbers = vect.size() / size;
    int count = num_of_local_numbers;
    if (rank == 0) {
        int begin;
        local_vect = vector<double>(num_of_local_numbers);
        std::copy(vect.begin(), vect.begin() + num_of_local_numbers, local_vect.begin());

        for (int process_num = 1; process_num < size; process_num++) {
            begin = num_of_local_numbers * process_num;
            if (process_num == size - 1) {
                count = vect.size() - begin;
            }
            MPI_Send(&vect[0] + num_of_local_numbers * process_num, count, MPI_DOUBLE, process_num, 0, MPI_COMM_WORLD);
        }
        count = num_of_local_numbers;
    } else {
        MPI_Status st;
        if (rank == size - 1) {
            count = vect.size() - num_of_local_numbers * (size - 1);
        }
        local_vect = vector<double>(count);
        MPI_Recv(&local_vect[0], count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &st);
    }
    vector<double> result = linearRadixSort(local_vect);

    vector<int> active_process(size);
    vector<int> process_deltas_map(size);
    for (int i = 0; i < size - 1; i++) {
        active_process[i] = i;
        process_deltas_map[i] = num_of_local_numbers;
    }
    active_process[size - 1] = size - 1;
    process_deltas_map[size - 1] = vect.size() - num_of_local_numbers * (size - 1);

    while (static_cast<int>(active_process.size()) > 1) {
        int index_of_cur_process = std::find(active_process.begin(), active_process.end(), rank)
            - active_process.begin();

        if ((index_of_cur_process == static_cast<int>(active_process.size()) - 1) &&
            (static_cast<int>(active_process.size()) % 2)) {
            vector<int> tmp_active_proc;
            vector<int> tmp_process_deltas_map;
            for (int i = 1; i < static_cast<int>(active_process.size()); i += 2) {
                tmp_active_proc.push_back(active_process[i]);
                tmp_process_deltas_map.push_back(process_deltas_map[i]);
            }
            tmp_active_proc.push_back(active_process[index_of_cur_process]);
            tmp_process_deltas_map.push_back(process_deltas_map[index_of_cur_process]);

            for (int i = 0; i < static_cast<int>(tmp_active_proc.size()); i++) {
                if (tmp_active_proc[i] != rank) {
                    tmp_process_deltas_map[i] *= 2;
                }
            }
            active_process = tmp_active_proc;
            continue;
        }
        if (index_of_cur_process % 2 == 0) {
            MPI_Send(&result[0], count, MPI_DOUBLE, active_process[index_of_cur_process + 1], 0, MPI_COMM_WORLD);
            if (rank != 0) {
                return vector<double>();
            } else {
                active_process = vector<int>(1);
                process_deltas_map = vector<int>(1);
                active_process[0] = size - 1;
                process_deltas_map[0] = static_cast<int>(vect.size());
            }
        } else {
            MPI_Status st;

            vector<double> tmp(process_deltas_map[index_of_cur_process - 1]);
            MPI_Recv(&tmp[0], process_deltas_map[index_of_cur_process - 1], MPI_DOUBLE,
                active_process[index_of_cur_process - 1], 0, MPI_COMM_WORLD, &st);

            result = merge(result, tmp);
            count = static_cast<int>(result.size());

            vector<int> tmp_active_proc;
            vector<int> tmp_process_deltas_map;
            for (int i = 1; i < static_cast<int>(active_process.size()); i += 2) {
                tmp_active_proc.push_back(active_process[i]);
                tmp_process_deltas_map.push_back(process_deltas_map[i]);
            }
            if ((static_cast<int>(active_process.size()) != 1) &&
                (static_cast<int>(active_process.size()) % 2)) {
                tmp_active_proc.push_back(active_process[static_cast<int>(active_process.size()) - 1]);
                tmp_process_deltas_map.push_back(process_deltas_map[static_cast<int>(active_process.size()) - 1]);
            }

            for (int i = 0; i < static_cast<int>(tmp_active_proc.size()); i++) {
                if (tmp_active_proc[i] != rank) {
                    tmp_process_deltas_map[i] *= 2;
                } else {
                    tmp_process_deltas_map[i] = count;
                }
            }
            active_process = tmp_active_proc;
            process_deltas_map = tmp_process_deltas_map;
        }
    }
    if (rank == active_process[0]) {
        MPI_Send(&result[0], count, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        vector<double> tmp(static_cast<int>(vect.size()));
        MPI_Recv(&tmp[0], static_cast<int>(vect.size()),
            MPI_DOUBLE, active_process[0], 0, MPI_COMM_WORLD, &status);
        return tmp;
    }
    return vector<double>();
}
