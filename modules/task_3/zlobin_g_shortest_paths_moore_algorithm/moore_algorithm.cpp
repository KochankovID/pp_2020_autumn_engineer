  // Copyright 2020 Zlobin George
#include <mpi.h>
#include <cstring>
#include <algorithm>
#include <random>
#include "../../../modules/task_3/zlobin_g_shortest_paths_moore_algorithm/moore_algorithm.h"


void fillGraphWithRandomValues(int* graph_to_fill, int size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> uniform_dis(1, MAX_VALUE);

    for (int i = 0; i < size; i++) {
        graph_to_fill[i * size + i] = 0;
        for (int j = i + 1; j < size; j++) {
            int weight = uniform_dis(gen);
            graph_to_fill[i * size + j] = (static_cast<double>(weight) / MAX_VALUE > EDGE_PROB) ?
                MAX_VALUE * (size + 1) : weight;
            weight = uniform_dis(gen);
            graph_to_fill[j * size + i] = (static_cast<double>(weight) / MAX_VALUE > EDGE_PROB) ?
                MAX_VALUE * (size + 1) : weight;
        }
    }
}

void getSequentialMooreAlgorithm(int* graph, int size, int* shortest_ways, int begin_vertex) {
    int max_value = *std::max_element(graph, graph + size * size);
    for (int i = 0; i < size; i++) {
        shortest_ways[i] = max_value * (size + 1);
    }
    shortest_ways[begin_vertex] = 0;
    int was_changed;
    for (int i = 0; i < size - 1; i++) {
        was_changed = 0;
        for (int v = 0; v < size; v++) {
            for (int u = 0; u < size; u++) {
                if (shortest_ways[v] > shortest_ways[u] + graph[u * size + v]) {
                    shortest_ways[v] = shortest_ways[u] + graph[u * size + v];
                    was_changed = 1;
                }
            }
        }
        if (!was_changed) {
            break;
        }
    }
}

void getParallelMooreAlgorithm(int* graph, int size, int* shortest_ways, int begin_vertex) {
    int loc_begin, loc_end;

    int num_processes, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int delta = size / num_processes;
    loc_begin = delta * rank;
    loc_end = delta * (rank + 1);
    if (rank == num_processes - 1) {
        loc_end = size;
    }

    int* loc_shortest_ways = new int[size];
    int max_value = *std::max_element(graph, graph + size * size);
    for (int i = 0; i < size; i++) {
        loc_shortest_ways[i] = max_value * (size + 1);
    }
    loc_shortest_ways[begin_vertex] = 0;

    int was_changed;
    for (int i = 0; i < size - 1; i++) {
        was_changed = 0;
        for (int v = loc_begin; v < loc_end; v++) {
            for (int u = 0; u < size; u++) {
                if (loc_shortest_ways[v] > loc_shortest_ways[u] + graph[u * size + v]) {
                    loc_shortest_ways[v] = loc_shortest_ways[u] + graph[u * size + v];
                    was_changed = 1;
                }
            }
        }
        MPI_Allreduce(MPI_IN_PLACE, &was_changed, 1, MPI_INT, MPI_BOR, MPI_COMM_WORLD);
        if (!was_changed) {
            break;
        }
        MPI_Allreduce(MPI_IN_PLACE, &loc_shortest_ways[0], size, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    }

    if (rank == 0)
        memcpy(shortest_ways, loc_shortest_ways, size * sizeof(int));

    delete[] loc_shortest_ways;
}
