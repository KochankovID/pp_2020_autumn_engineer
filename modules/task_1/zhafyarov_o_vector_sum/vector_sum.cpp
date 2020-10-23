  // Copyright 2020 Zhafyarov Oleg
#include <mpi.h>
#include <vector>
#include <random>
#include <ctime>
#include "../../../modules/task_1/zhafyarov_o_vector_sum/vector_sum.h"

std::vector <double> getRandomVector(int Size) {
    std::vector<double> vectmp(Size);

    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    for (int i = 0; i < Size; i++) {
        vectmp[i] = gen() % 50;
    }

    return vectmp;
}

int getParallelOperations(std::vector <int> vec, int Size) {
    int ProcNum, ProcRank;
    int buffer = 0;
    int sum = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &ProcNum);
    MPI_Comm_rank(MPI_COMM_WORLD, &ProcRank);
    MPI_Status status;
    int SizeForProc = Size / ProcNum;
    int Div = Size % ProcNum;

    if (ProcRank == 0) {
        for (int i = 1; i < ProcNum; i++) {
            MPI_Send(&vec[0] + i * SizeForProc + Div, SizeForProc, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

        for (int i = 0; i < Div + SizeForProc; i++) {
            buffer += vec[i];
        }
    } else {
        std::vector <int> vec(SizeForProc);

        MPI_Recv(&vec[0], SizeForProc, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

        for (int i = 0; i < SizeForProc; i++) {
            buffer += vec[i];
        }
    }

    MPI_Reduce(&buffer, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    vec.clear();

    return sum;
}

int getSequentialOperations(std::vector<int> vec) {
    int sum = 0;
    const int Size = vec.size();
    for (int i = 0; i < Size; i++) {
        sum += vec[i];
    }

    return sum;
}
