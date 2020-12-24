// Copyright 2020 Gavrilov Nikita
#include <mpi.h>
#include <utility>
#include <algorithm>
#include <vector>
#include <cmath>
#include "../../modules/task_3/gavrilov_n_batcher_merging_sort/BatcherMergingSort.h"

#define MY_INT_MAX 2147483647

int8_t GetDigit(int value, uint8_t digidNum, uint8_t rang) {
    if (rang <= 1)
        throw "rang must be more than 1";

    int powRang = static_cast<int>(pow(rang, digidNum));
    int powRangNext = static_cast<int>(pow(rang, digidNum + 1.0));

    return value % powRangNext / powRang;
}
uint8_t GetNumOfDigits(int value, uint8_t rang) {
    if (rang <= 1)
        throw "rang must be more than 1";

    int length = 1;
    while (value /= rang)
        length++;

    return length;
}
void ConcatVectors(const std::vector<std::vector<int>>& from, std::vector<int>* dest) {
    for (size_t i = 0; i < from.size(); i++) {
        for (size_t j = 0; j < from[i].size(); j++) {
            dest->push_back(from[i][j]);
        }
    }
}
void SortByDigid(std::vector<int>* data, uint8_t digitNum, uint8_t rang) {
    if (rang <= 1)
        throw "rang must be more than 1";
    std::vector<std::vector<int>> vectorsByDigits(rang * 2 - 1);

    for (size_t i = 0; i < data->size(); i++) {
        int8_t digit = GetDigit((*data)[i], digitNum, rang);
        vectorsByDigits[digit + rang - 1].push_back((*data)[i]);
    }

    data->clear();
    ConcatVectors(vectorsByDigits, data);
}
void SortByNumPlace(std::vector<int>* data, uint8_t rang) {
    if (rang <= 1)
        throw "rang must be more than 1";

    std::vector<std::vector<int>> vectorsByDigits(rang * 2 - 1);
    uint8_t maxNumOfDigits = 0;

    // it is the same with SortByDigid, but we need to count maxNumOfDigits,
    // we can do this because number has 1 digit as minimum
    // bad part of this - we write same code
    // it makes sort faster by data.size() operations
    for (size_t i = 0; i < data->size(); i++) {
        int8_t digit = GetDigit((*data)[i], 0, rang);
        vectorsByDigits[digit + rang - 1].push_back((*data)[i]);

        uint8_t numOfDigits = GetNumOfDigits((*data)[i], rang);
        maxNumOfDigits = std::max(maxNumOfDigits, numOfDigits);
    }

    data->clear();
    ConcatVectors(vectorsByDigits, data);

    for (uint8_t i = 1; i < maxNumOfDigits; i++) {
        SortByDigid(data, i, rang);
    }
}

void SplitEvenOdd(const std::vector<int>& from, std::vector<int>* destEven, std::vector<int>* destOdd) {
    for (size_t i = 0; i < from.size() / 2; i++) {
        destEven->push_back(from[i * 2]);
        destOdd->push_back(from[i * 2 + 1]);
    }
    if (from.size() % 2 == 1) {
        destEven->push_back(from.back());
    }
}
void BatcherMerge(std::vector<int> procsLeft, std::vector<int> procsRight, std::vector<std::pair<int, int>>* comps) {
    int procsCount = procsLeft.size() + procsRight.size();
    if (procsCount <= 1) {
        return;
    } else if (procsCount == 2) {
        comps->push_back(std::pair<int, int>(procsLeft[0], procsRight[0]));
        return;
    }

    std::vector<int> procsLeftOdd, procsLeftEven;
    std::vector<int> procsRightOdd, procsRightEven;
    std::vector<int> procsResult;

    SplitEvenOdd(procsLeft, &procsLeftEven, &procsLeftOdd);
    SplitEvenOdd(procsRight, &procsRightEven, &procsRightOdd);

    BatcherMerge(procsLeftOdd, procsRightOdd, comps);
    BatcherMerge(procsLeftEven, procsRightEven, comps);

    std::vector<std::vector<int>> vecs{ procsLeft, procsRight };
    ConcatVectors(vecs, &procsResult);

    for (size_t i = 1; i + 1 < procsResult.size(); i += 2) {
        comps->push_back(std::pair<int, int>(procsResult[i], procsResult[i + 1]));
    }
}
void BatcherSplitNMerge(std::vector<int> procs, std::vector<std::pair<int, int>>* comps) {
    if (procs.size() <= 1) {
        return;
    }

    std::vector<int> procs_up(procs.begin(), procs.begin() + procs.size() / 2);
    std::vector<int> procs_down(procs.begin() + procs.size() / 2, procs.end());

    BatcherSplitNMerge(procs_up, comps);
    BatcherSplitNMerge(procs_down, comps);
    BatcherMerge(procs_up, procs_down, comps);
}
std::vector<std::pair<int, int>> Batcher(int procCount) {
    std::vector<int> procs(procCount);
    for (int i = 0; i < procCount; i++) {
        procs[i] = i;
    }

    std::vector<std::pair<int, int>> comps;
    BatcherSplitNMerge(procs, &comps);

    return comps;
}

void Sort(std::vector<int>* data) {
    size_t realSize = data->size();

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int maxSize = std::min(size, static_cast<int>(data->size()));
    // calculating comparers
    std::vector<std::pair<int, int>> comps = Batcher(maxSize);

    // calculating how many of data will owns each proc
    size_t countPerProc;
    if (rank == 0) {
        while (data->size() % size != 0) {
            data->push_back(MY_INT_MAX);
        }
        countPerProc = data->size() / size;
    }
    MPI_Bcast(&countPerProc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // splitting data by procs
    std::vector<int> localData;
    if (rank == 0) {
        for (int i = 1; i < maxSize; i++) {
            MPI_Send(data->data() + (countPerProc * i), countPerProc, MPI_INT, i, 1, MPI_COMM_WORLD);
        }
        localData = std::vector<int>(data->begin(), data->begin() + countPerProc);
    } else if (rank < maxSize) {
        localData = std::vector<int>(countPerProc);
        MPI_Status st;
        MPI_Recv(localData.data(), countPerProc, MPI_INT, 0, 1, MPI_COMM_WORLD, &st);
    }

    // sort each proc's data
    SortByNumPlace(&localData);

    if (size > 1) {
        // making all comparers
        for (size_t i = 0; i < comps.size(); i++) {
            if (rank == comps[i].first) {
                MPI_Send(localData.data(), countPerProc, MPI_INT, comps[i].second, 2, MPI_COMM_WORLD);
                MPI_Status st;
                MPI_Recv(localData.data(), countPerProc, MPI_INT, comps[i].second, 3, MPI_COMM_WORLD, &st);
            } else if (rank == comps[i].second) {
                MPI_Status st;

                std::vector<int> recvData(countPerProc);
                MPI_Recv(recvData.data(), countPerProc, MPI_INT, comps[i].first, 2, MPI_COMM_WORLD, &st);

                std::vector<int> tmp(countPerProc * 2);

                for (size_t j = 0, k = 0, l = 0; l < countPerProc * 2; l++) {
                    if (j < countPerProc && (k >= countPerProc || localData[j] < recvData[k]))
                        tmp[l] = localData[j++];
                    else
                        tmp[l] = recvData[k++];
                }
                MPI_Send(tmp.data(), countPerProc, MPI_INT, comps[i].first, 3, MPI_COMM_WORLD);
                for (size_t j = 0; j < countPerProc; j++) {
                    localData[j] = tmp[j + countPerProc];
                }
            }
        }

        // recieving data from all procs to main one
        if (rank == 0) {
            for (int i = 1; i < maxSize; i++) {
                MPI_Status st;
                MPI_Recv(data->data() + (countPerProc * i), countPerProc, MPI_INT, i, 4, MPI_COMM_WORLD, &st);
            }
        } else if (rank < maxSize) {
            MPI_Send(localData.data(), countPerProc, MPI_INT, 0, 4, MPI_COMM_WORLD);
        }
    }

    if (rank == 0) {
        // copying data from main to itself
        std::copy(localData.begin(), localData.end(), data->begin());

        // erasing added vars;
        data->erase(data->begin() + realSize, data->end());
    }
}
