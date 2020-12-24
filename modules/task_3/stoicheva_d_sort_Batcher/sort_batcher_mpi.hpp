// Copyright 2020 Stoicheva Darya
#ifndef MODULES_TASK_2_STOICHEVA_D_SORT_BATCHER_MPI_H_
#define MODULES_TASK_2_STOICHEVA_D_SORT_BATCHER_MPI_H_

#include <mpi.h>
#include <vector>
#include <string>
#include <iostream>

template<typename T>
extern std::vector<T> getSequentialOperations(const std::vector<T> source);

template<typename T>
extern std::vector<T> getParallelOperations(const std::vector<T> source);

template <typename T>
extern int compare(const void * a, const void * b);

void print_vector(const std::vector<int> vector, const size_t count, const std::string line_prefix = {},
    const std::string vector_label = {}, bool one_line = false, std::ostream &out = std::cout);

template<typename T>
void print_vector(const std::vector<T> vector, const size_t count, const std::string line_prefix = {},
    const std::string vector_label = {}, bool one_line = false, std::ostream &out = std::cout);

// GLOBAL VARIABLES
extern int my_mpiRoot;

#define DEBUG_PRINT

#endif  // MODULES_TASK_2_STOICHEVA_D_SORT_BATCHER_MPI_H_
