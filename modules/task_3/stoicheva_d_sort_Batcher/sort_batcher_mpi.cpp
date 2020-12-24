// Copyright 2020 Stoicheva Darya
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <random>
#include <ctime>
#include <cmath>
#include <limits>
#include <string>
#include <algorithm>
#include "sort_batcher_mpi.hpp"


template<typename result, typename source>
result pointer_cast(source *v) {
    return static_cast<result>(static_cast<void*>(v));
}


template<typename T>
std::string vector_as_string(const std::vector<T> vector, const size_t count, const std::string line_prefix,
    const std::string vector_label, bool one_line) {
#ifdef DEBUG_PRINT
    std::stringstream buffer;

    std::string endl = one_line ? "" : "\n";
    std::string prefix = line_prefix;
    std::string label = vector_label.empty() ? "Vector" : vector_label;
    size_t items_to_print = count == 0 ? vector.size() : std::min(count, vector.size());

    buffer << prefix + label + ": {" << endl;
    if (one_line) {
        prefix = "";
    } else {
        buffer << prefix << "  ";
    }
    for (size_t i = 0; i < items_to_print; i++) {
        buffer << vector[i] << ", ";
    }
    buffer << (items_to_print < vector.size() ? "..." : "") << endl << prefix << "}" << endl;
    return buffer.str();
#endif
}

std::string vector_as_string(const std::vector<int> vector, const size_t count, const std::string line_prefix,
    const std::string vector_label, bool one_line) {
    return vector_as_string<int>(vector, count, line_prefix, vector_label, one_line);
}


std::string vector_as_string(const std::vector<double> vector, const size_t count, const std::string line_prefix,
    const std::string vector_label, bool one_line) {
    return vector_as_string<double>(vector, count, line_prefix, vector_label, one_line);
}


std::string vector_as_string(const std::vector<float> vector, const size_t count, const std::string line_prefix,
    const std::string vector_label, bool one_line) {
    return vector_as_string<float>(vector, count, line_prefix, vector_label, one_line);
}


template<typename result, typename source>
result pointer_cast(const source *v) {
    return static_cast<result>(const_cast<void *>(static_cast<const void*>(v)));
}


void print_vector(const std::vector<int> vector, const size_t count, const std::string line_prefix,
    const std::string vector_label, bool one_line, std::ostream &out) {
    print_vector<int>(vector, count, line_prefix, vector_label, one_line, out);
}


void print_vector(const std::vector<float> vector, const size_t count, const std::string line_prefix,
    const std::string vector_label, bool one_line, std::ostream &out) {
    print_vector<float>(vector, count, line_prefix, vector_label, one_line, out);
}


void print_vector(const std::vector<double> vector, const size_t count, const std::string line_prefix,
    const std::string vector_label, bool one_line, std::ostream &out) {
    print_vector<double>(vector, count, line_prefix, vector_label, one_line, out);
}


template<typename T>
void print_vector(const std::vector<T> vector, const size_t count, const std::string line_prefix,
    const std::string vector_label, bool one_line, std::ostream &out) {
#ifdef DEBUG_PRINT
    out << vector_as_string<T>(vector, count, line_prefix, vector_label, one_line) << std::flush;
#endif
}


template<typename T>
std::vector<T> getSequentialOperations(const std::vector<T> source) {
    std::vector<T> sorted(source);
    std::sort(sorted.begin(), sorted.end());

    return sorted;
}
template std::vector<int> getSequentialOperations<int>(const std::vector<int> source);
template std::vector<float>  getSequentialOperations<float>(const std::vector<float> source);
template std::vector<double>  getSequentialOperations<double>(const std::vector<double> source);


template <typename T>
int compare(const void * a, const void * b) {
    T *aa = reinterpret_cast<T *>(const_cast<void *>(a));
    T *bb = reinterpret_cast<T *>(const_cast<void *>(b));

    if (*aa > *bb) {
        return 1;
    } else if (*aa < *bb) {
        return -1;
    }
    return 0;
}


MPI_Datatype get_datatype(int value) {
    return MPI_INT;
}

MPI_Datatype get_datatype(float value) {
    return MPI_FLOAT;
}

MPI_Datatype get_datatype(double value) {
    return MPI_DOUBLE;
}


void dbg_print(int rank, std::string message, bool add_nl = false) {
#ifdef DEBUG_PRINT
    std::string smessage = "[" + std::to_string(rank) + "] " + message;
    if (add_nl) {
        std::cout << smessage << std::endl;
    } else {
        std::cout << smessage << std::flush;
    }
#endif
}


template<typename T>
void merge_lowest(std::vector<T> *target, std::vector<T> buffer) {
    std::vector<T> lowest(target->size());
    size_t i1, i2, count;
    i1 = 0, i2 = 0, count = 0;
    while (count < target->size()) {
        if (target->at(i1) < buffer[i2]) {
            lowest[count] = target->at(i1++);
        } else {
            lowest[count] = buffer[i2++];
        }
        count++;
    }
    target->assign(lowest.begin(), lowest.end());
}


template<typename T>
void merge_highest(std::vector<T> *target, std::vector<T> buffer) {
    std::vector<T> highest(target->size());
    int i1, i2, count;
    i1 = i2 = count = target->size() - 1;
    while (count >= 0) {
        if (target->at(i1) > buffer[i2]) {
            highest[count] = target->at(i1--);
        } else {
            highest[count] = buffer[i2--];
        }
        count--;
    }
    target->assign(highest.begin(), highest.end());
}


size_t get_rank(MPI_Comm comm = MPI_COMM_WORLD) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

size_t get_world_size(MPI_Comm comm = MPI_COMM_WORLD) {
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    return world_size;
}

template<typename T>
void merge(MPI_Comm comm, std::vector<T> *data, std::vector<T> buffer, int count_items, int low, int high) {
    int rank = get_rank(comm);
    int world_size = get_world_size(comm);
    MPI_Datatype datatype = get_datatype(data->at(0));
    MPI_Status recv_status;

    if (rank == low && high < world_size) {
        MPI_Send(data->data(), count_items, datatype, high, 0, comm);
        MPI_Recv(buffer.data(), count_items, datatype, high, 0, comm, &recv_status);
        merge_lowest<T>(data, buffer);
    } else if (rank == high) {
        MPI_Send(data->data(), count_items, datatype, low, 0, comm);
        MPI_Recv(buffer.data(), count_items, datatype, low, 0, comm, &recv_status);
        merge_highest<T>(data, buffer);
    }
}


// https://en.wikipedia.org/wiki/Batcher_odd%E2%80%93even_mergesort
template<typename T>
void batcher_sort(MPI_Comm comm, std::vector<T> *data, int count_items) {
    std::vector<T> recv_buf(count_items);
    int size = get_world_size(comm);

    for (int p = 1; p < size; p *= 2) {
        for (int k = p; k >= 1; k /= 2) {
            for (int j = k % p; j <= size - 1 - k; j += k + k) {
                for (int i = 0; i <= k - 1; i++) {
                    if (std::floor((i + j) / (2 * p)) == std::floor((i + j + k) / (2 * p))) {
                        merge<T>(comm, data, recv_buf, count_items, i + j, i + j + k);
                    }
                }
            }
        }
    }
}


template<typename T>
std::vector<T> getParallelOperations(const std::vector<T> source) {
    int world_size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm comm = MPI_COMM_WORLD;

    size_t count_items_per_proc = source.size() / world_size;
    if (count_items_per_proc * world_size < source.size())
        count_items_per_proc++;

    T max_value = std::numeric_limits<T>::max();
    size_t count_extra_items = count_items_per_proc * world_size - source.size();

    std::vector<T> result{};

    if (rank == 0) {
        result.assign(source.begin(), source.end());
    }

    if (count_items_per_proc < 3 || world_size == 1) {
        if (rank == 0) {
            qsort(result.data(), result.size(), sizeof(T), compare<T>);
            return result;
        }
        return result;
    }

    if (rank == 0 && count_extra_items > 0) {
        std::vector<T> extend(count_extra_items, max_value);
        result.insert(result.end(), extend.begin(), extend.end());
        // dbg_print(rank, vector_as_string(result, 50, "", "source extended", true), true);
    }

    // 1. Scatter source data
    // 1. Scatter source data
    std::vector<T> proc_source(count_items_per_proc);
    MPI_Datatype datatype = get_datatype(proc_source[0]);
    MPI_Scatter(result.data(), count_items_per_proc, datatype,
                proc_source.data(), count_items_per_proc, datatype, 0, comm);

    // 2. Each process make qsort of local data
    qsort(proc_source.data(), proc_source.size(), sizeof(T), compare<T>);

    // 3. Batcher sort
    batcher_sort(comm, &proc_source, count_items_per_proc);

    MPI_Barrier(comm);

    // 4. Gather sorted data
    MPI_Gather(proc_source.data(), count_items_per_proc, datatype,
               result.data(), count_items_per_proc, datatype, 0, comm);

    if (rank == 0 && count_extra_items > 0) {
        result.resize(source.size());
    }

    return result;
}
template std::vector<int> getParallelOperations<int>(const std::vector<int> source);
template std::vector<float>  getParallelOperations<float>(const std::vector<float> source);
template std::vector<double>  getParallelOperations<double>(const std::vector<double> source);
