// Copyright 2020 Eremina Alena
#ifndef MODULES_TASK_3_EREMINA_A_CANNON_CANNON_H_
#define MODULES_TASK_3_EREMINA_A_CANNON_CANNON_H_

#include <mpi.h>
#include <vector>
#include <string>

std::vector<double> getParallelMultiply(std::vector<double> a, std::vector<double> b, int size);
std::vector<double> getSequentialMultiply(std::vector<double> a, std::vector<double> b, int size);
void CartesianComm(int block_size, MPI_Comm *BlocksComm);
std::vector<double> getRandomMatrix(int size);

#endif  // MODULES_TASK_3_EREMINA_A_CANNON_CANNON_H_
