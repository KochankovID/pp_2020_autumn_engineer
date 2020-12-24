// Copyright 2020 Egorov Kirill
#include <mpi.h>
#include <vector>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <iostream>
#include "../../../modules/task_3/egorov_k_batcher_sort/batcher_sort.h"


std::vector<int> getRandomVector(int sz) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<int> vec(sz);
    for (int i = 0; i < sz; i++) { vec[i] = gen() % 100; }
    return vec;
}

void print(std::vector <int> const &a) {
    std::cout << "The vector elements are : ";

    for (unsigned int i = 0; i < a.size(); i++)
         std::cout << a.at(i) << ' ';

    std::cout << std::endl;
}

void compEx(std::vector<int> *vec, int i, int j) {
    std::vector<int>::iterator it = vec->begin();

    if (vec->at(j) < vec->at(i)) {
        std::advance(it, i);
        vec->insert(it, vec->at(j));
        it = vec->begin();
        std::advance(it, j + 1);
        vec->erase(it);

        it = vec->begin();
    }
}

void sh(std::vector<int> *vec, int l, int r) {
    std::vector<int> aux;
    int m = (l + r) / 2;

    if (vec->size() % 2 != 0) {
        int lValue = vec->at(vec->size() - 1);

        vec->erase(vec->end() - 1);

        for (int j = l; j < (r - 1) / 2; j++) {
            aux.push_back(vec->at(j));
            aux.push_back(vec->at(j + m));
        }

        aux.push_back(lValue);
    } else {
        for (int j = l; j < r / 2; j++) {
            aux.push_back(vec->at(j));
            aux.push_back(vec->at(j + m));
        }
    }

    vec->erase(vec->begin(), vec->end());
    vec->shrink_to_fit();

    for (int i = l; i < r; i++) {
        vec->push_back(aux[i]);
    }
}

void uns(std::vector<int> *vec, int l, int r) {
    std::vector<int> auxLeft, auxRight;

    if (vec->size() % 2 != 0) {
        int lValue = vec->at(vec->size() - 1), lPos = vec->size() - 1;

        vec->erase(vec->end() - 1);

        for (int j = l; j < r - 1; j += 2) {
            auxLeft.push_back(vec->at(j));
            auxRight.push_back(vec->at(j + 1));
        }

        if (lPos % 2 == 0)
            auxLeft.push_back(lValue);
        else
            auxRight.push_back(lValue);
    } else {
        for (int j = l; j < r; j += 2) {
            auxLeft.push_back(vec->at(j));
            auxRight.push_back(vec->at(j + 1));
        }
    }

    vec->erase(vec->begin(), vec->end());
    vec->shrink_to_fit();

    vec->reserve(auxLeft.size() + auxRight.size());
    vec->insert(vec->end(), auxLeft.begin(), auxLeft.end());
    vec->insert(vec->end(), auxRight.begin(), auxRight.end());
}

bool simpleCheck(std::vector<int> const &vec) {
    for (unsigned int i = 0; i < vec.size() - 1; i++)
        if (vec[i + 1] < vec[i])
           return false;

    return true;
}

void simpleSort(std::vector<int> *vec) {
    while (!simpleCheck(*vec)) {
        for (unsigned int i = 0; i < vec->size() - 1; i++) {
            compEx(vec, i, i + 1);
        }
    }
}

void BatcherSort(std::vector<int> *vec) {
    if (vec->size() == 2)
        compEx(vec, 0, 1);

    if (vec->size() == 1)
        return;

    std::vector<int> vecAuxLeft, vecAuxRight;
    vecAuxLeft.reserve(vec->size() / 2);
    vecAuxRight.reserve(vec->size() / 2);

    uns(vec, 0, vec->size());

    vecAuxLeft.insert(vecAuxLeft.end(), vec->begin(), vec->begin() + vec->size() / 2);
    vecAuxRight.insert(vecAuxRight.end(), vec->begin() + vec->size() / 2, vec->end());

    BatcherSort(&vecAuxLeft);
    BatcherSort(&vecAuxRight);

    vec->erase(vec->begin(), vec->end());
    vec->shrink_to_fit();

    vec->insert(vec->end(), vecAuxLeft.begin(), vecAuxLeft.end());
    vec->insert(vec->end(), vecAuxRight.begin(), vecAuxRight.end());

    sh(vec, 0, vec->size());

    if (vec->size() != 2)
        simpleSort(vec);
}

void BatcherSortParallel(std::vector<int> *vec) {
    int numtasks, rank, elemCount, rmPart, count, localCount = 0;
    std::vector<int> locVec, auxLeft, auxRight;

    MPI_Status status;

    MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int check = 2;
    count = 0;

    while (check <= numtasks) {
        check = check * 2;
        localCount++;
    }

    numtasks = check / 2;

    MPI_Bcast(vec->data(), vec->size(), MPI_INT, 0, MPI_COMM_WORLD);

    if (rank < numtasks) {
        elemCount = vec->size() / numtasks;
        rmPart = vec->size() % numtasks;

        if (rank == 0) {
            auxLeft.insert(auxLeft.end(), vec->begin(), vec->begin() + elemCount + rmPart);
        } else {
            auxLeft.insert(auxLeft.end(), vec->begin() + elemCount * rank + rmPart,
            vec->begin() + elemCount * rank + rmPart + elemCount);
        }

        BatcherSort(&auxLeft);

        for (count = localCount; count > 0; count--) {
            int currentProcesses = pow(2, count);
            int half = currentProcesses / 2;
            if (rank < currentProcesses) {
                if (rank > half - 1) {
                    elemCount = auxLeft.size();
                    MPI_Send(&elemCount, 1, MPI_INT, rank - half, 1, MPI_COMM_WORLD);
                    MPI_Send(auxLeft.data(), auxLeft.size(), MPI_INT, rank - half, 1, MPI_COMM_WORLD);
                } else {
                    MPI_Recv(&elemCount, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                    auxRight.resize(elemCount);
                    MPI_Recv(auxRight.data(), elemCount, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

                    locVec.erase(locVec.begin(), locVec.end());
                    locVec.shrink_to_fit();

                    locVec.insert(locVec.end(), auxLeft.begin(), auxLeft.end());
                    locVec.insert(locVec.end(), auxRight.begin(), auxRight.end());
                    sh(&locVec, 0, locVec.size());

                    BatcherSort(&locVec);
                    auxLeft.erase(auxLeft.begin(), auxLeft.end());
                    auxLeft.shrink_to_fit();
                    auxLeft.insert(auxLeft.end(), locVec.begin(), locVec.end());
                }
            }
        }
    }

    if (rank == 0) {
        vec->erase(vec->begin(), vec->end());
        vec->shrink_to_fit();
        vec->insert(vec->end(), locVec.begin(), locVec.end());
    }
}
