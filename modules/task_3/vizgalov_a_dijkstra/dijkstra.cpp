// Copyright 2020 Vizgalov Anton
#include <mpi.h>

#include <limits>
#include <random>
#include <vector>

#include "../../../modules/task_3/vizgalov_a_dijkstra/dijkstra.h"


std::vector<int> generateRandomGraph(int size) {
    std::vector<int> graph;

    std::random_device dev;
    std::mt19937 randomGenerator(dev());
    std::uniform_int_distribution<int> randomValue(1, 100);

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            if (i < j) {
                graph.push_back(randomValue(randomGenerator));
            }
            if (i == j) {
                graph.push_back(0);
            }
            if (i > j) {
                graph.push_back(graph[j * size + i]);
            }
        }
    }

    return graph;
}

std::vector<int> dijkstraSequential(std::vector<int> graph, int startVertexIdx) {
    const int numOfVertices = static_cast<int>(sqrt(graph.size()));
    std::vector<int> visited(numOfVertices, 0);
    std::vector<int> distance(numOfVertices, std::numeric_limits<int>::max());
    distance[startVertexIdx] = 0;

    for (int i = 0; i < numOfVertices - 1; ++i) {
        int vertex = -1;

        for (int j = 0; j < numOfVertices; ++j) {
            if (!visited[j] && (vertex == -1 || distance[j] < distance[vertex])) {
                vertex = j;
            }
        }

        if (distance[vertex] == std::numeric_limits<int>::max()) {
            break;
        }
        visited[vertex] = 1;

        for (int j = 0; j < numOfVertices; ++j) {
            if (graph[vertex * numOfVertices + j] != 0
                && distance[vertex] + graph[vertex * numOfVertices + j] < distance[j]) {
                distance[j] = distance[vertex] + graph[vertex * numOfVertices + j];
            }
        }
    }

    return graph;
}

std::vector<int> dijkstraParallel(std::vector<int> graph, int startVertexIdx) {
    int commSize, procRank;
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &procRank);

    const int numOfVertices = static_cast<int>(sqrt(graph.size()));
    const int verticesPerProc = numOfVertices / commSize;
    const int leftover = numOfVertices % commSize;
    std::vector<int> visited(numOfVertices, 0);
    std::vector<int> distance(numOfVertices, std::numeric_limits<int>::max());
    distance[startVertexIdx] = 0;
    struct Vertex { int idx; int val; };

    std::vector<int> localGraph(numOfVertices * verticesPerProc);
    MPI_Scatter(&graph[leftover * numOfVertices], verticesPerProc * numOfVertices, MPI_INT, &localGraph[0],
                verticesPerProc * numOfVertices, MPI_INT, 0, MPI_COMM_WORLD);

    if (procRank == 0) {
        localGraph.insert(localGraph.begin(), graph.begin(), graph.begin() + leftover * numOfVertices);
    }

    int localGraphSize = localGraph.size() / numOfVertices;
    int delta = (procRank == 0) ? 0 : leftover + verticesPerProc * procRank;
    Vertex localVertex, globalVertex;
    for (int i = 0; i < numOfVertices - 1; ++i) {
        localVertex.val = -1;
        localVertex.idx = -1;

        for (int j = delta; j < localGraphSize + delta; ++j) {
            if (!visited[j] && (localVertex.idx == -1 || distance[j] < distance[localVertex.idx])) {
                localVertex.idx = j;
                localVertex.val = distance[localVertex.idx];
            }
        }

        if (localVertex.idx == -1) {
            localVertex.val = std::numeric_limits<int>::max();
        }

        MPI_Allreduce(&localVertex, &globalVertex, 1, MPI_2INT, MPI_MINLOC, MPI_COMM_WORLD);

        if (globalVertex.idx == -1 || distance[globalVertex.idx] == std::numeric_limits<int>::max()) {
            break;
        }
        visited[globalVertex.idx] = 1;

        for (int j = 0; j < localGraphSize; ++j) {
            if (localGraph[globalVertex.idx + numOfVertices * j] != 0
                && distance[globalVertex.idx] + localGraph[globalVertex.idx + numOfVertices * j]
                   < distance[j + delta]) {
                distance[j + delta] = distance[globalVertex.idx] + localGraph[globalVertex.idx + numOfVertices * j];
            }
        }

        if (procRank == 0) {
            std::vector<int> receivedResult(verticesPerProc);
            distance.resize(leftover + verticesPerProc);

            for (int i = 1; i < commSize; ++i) {
                MPI_Recv(&receivedResult[0], verticesPerProc, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                distance.insert(distance.end(), receivedResult.begin(), receivedResult.end());
            }
        } else {
            for (int i = 1; i < commSize; ++i) {
                if (procRank == i) {
                    MPI_Send(&distance[leftover + procRank * verticesPerProc], verticesPerProc, MPI_INT, 0, 0,
                             MPI_COMM_WORLD);
                }
            }
        }

        MPI_Bcast(&distance[0], distance.size(), MPI_INT, 0, MPI_COMM_WORLD);
    }

    return graph;
}
