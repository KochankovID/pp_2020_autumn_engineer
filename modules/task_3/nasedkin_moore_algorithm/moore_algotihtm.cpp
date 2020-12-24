// Copyright 2020 Nasedkin Nikita
#include <mpi.h>
#include <iostream>
#include <random>
#include <ctime>
#include <vector>
#include "../../modules/task_3/nasedkin_moore_algorithm/moore_algotihtm.h"
const int big_val = 1000000;

std::vector<int> GetRandomGraph(int size) {
  std::mt19937 rand_r;
  rand_r.seed(static_cast<unsigned int>(time(0)));
  std::vector<int> graph(size * size);

  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      graph[j + (i * size)] = (rand_r() % 100);
      if (graph[j + (i * size)] == 0) {
        graph[j + (i * size)] = big_val;
      }
      if (j == i) {
        graph[j + (i * size)] = 0;
      }
    }
  }
  return graph;
}

std::vector<int> MooreSequential(std::vector<int>* graph, int start) {
  int size = static_cast<int>(sqrt(static_cast<int>(graph->size())));
  std::vector<int> dist(size, big_val);
  dist[start] = 0;
  for (int i = 0; i < size - 1; i++) {
    for (int j = 0; j < size; j++) {
      for (int k = 0; k < size; k++) {
        if (dist[j] < big_val && ((*graph)[k + (j * size)] < big_val)) {
          if (dist[k] > dist[j] + (*graph)[k + (j * size)]) {
            dist[k] = dist[j] + (*graph)[k + (j * size)];
          }
        }
      }
    }
  }
  return dist;
}

std::vector<int> MooreParallel(std::vector<int>* graph, int start) {
  int size = static_cast<int>(sqrt(static_cast<int>(graph->size())));
  std::vector<int> dist(size, big_val);
  dist[start] = 0;

  int proc_rank, proc_size, root_proc = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_size);

  int step = size / proc_size;
  std::vector<int> local_dist((step + (proc_rank < (size% proc_size) ? 1 : 0)), big_val);
  std::vector<int> local_gr((step + (proc_rank < (size% proc_size) ? 1 : 0)) * size);

  std::vector<int> scounts(proc_size);
  std::vector<int> displs(proc_size);
  std::vector<int> scounts_dist(proc_size);
  std::vector<int> displs_dist(proc_size);

  displs[0] = displs_dist[0] = 0;
  for (int i = 0; i < proc_size; i++) {
    scounts[i] = (step + (i < (size% proc_size) ? 1 : 0)) * size;
    scounts_dist[i] = step + (i < size% proc_size ? 1 : 0);
    if (i > 0) {
      displs[i] = (displs[(i - 1)] + scounts[(i - 1)]);
      displs_dist[i] = displs_dist[(i - 1)] + scounts_dist[(i - 1)];
    }
  }

  for (int i = 0; i < proc_size - 1; ++i)
    if (start >= displs[i] / size)
      root_proc++;
  if (proc_rank == root_proc)
    local_dist[start - (displs[proc_rank] / size)] = 0;

  std::vector<int> send(size * size);
  if (proc_rank == 0) {
    for (int i = 0; i < size; i++)
      for (int j = 0; j < size; j++)
        send[(i * size) + j] = (*graph)[(j * size) + i];
  }

  MPI_Scatterv(send.data(), scounts.data(), displs.data(),
    MPI_INT, &local_gr[0], local_gr.size(), MPI_INT, 0, MPI_COMM_WORLD);

  for (int i = 0; i < size - 1; i++) {
    for (int k = 0; k < scounts_dist[proc_rank]; k++) {
      for (int j = 0; j < size; ++j)
        if ((dist[j] < big_val) && (local_gr[(k * size) + j] < big_val))
          if (local_dist[k] > dist[j] + local_gr[(k * size) + j])
            local_dist[k] = dist[j] + local_gr[(k * size) + j];
    }
    MPI_Allgatherv(local_dist.data(), (step + (proc_rank < size% proc_size ? 1 : 0)), MPI_INT, dist.data(),
      scounts_dist.data(), displs_dist.data(), MPI_INT, MPI_COMM_WORLD);
  }
  return dist;
}
