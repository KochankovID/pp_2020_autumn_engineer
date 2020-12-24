// Copyright 2020 Vizgalov Anton
#ifndef MODULES_TASK_3_VIZGALOV_A_DIJKSTRA_DIJKSTRA_H_
#define MODULES_TASK_3_VIZGALOV_A_DIJKSTRA_DIJKSTRA_H_

#include <vector>


std::vector<int> generateRandomGraph(int size);
std::vector<int> dijkstraSequential(std::vector<int> graph, int startVertexIdx);
std::vector<int> dijkstraParallel(std::vector<int> graph, int startVertexIdx);

#endif  // MODULES_TASK_3_VIZGALOV_A_DIJKSTRA_DIJKSTRA_H_
