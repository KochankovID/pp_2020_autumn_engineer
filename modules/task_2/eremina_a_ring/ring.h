// Copyright 2020 Eremina Alena
#ifndef MODULES_TASK_2_EREMINA_A_RING_RING_H_
#define MODULES_TASK_2_EREMINA_A_RING_RING_H_

#include <mpi.h>
#include <vector>

MPI_Comm topologyRing(int* order);
std::vector<int> getRandomVector(int size);

#endif  // MODULES_TASK_2_EREMINA_A_RING_RING_H_
