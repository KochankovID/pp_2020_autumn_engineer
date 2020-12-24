// Copyright 2018 Sharov Alexander
#include <mpi.h>
#include <vector>
#include <ctime>
#include<random>
#include <iostream>
#include <utility>
#include "../../../modules/task_3/sharov_a_shell/shell.h"

std::vector<int> sortshell(std::vector<int> a, int n) {
    int d, f, l;
    l = n;
    d = n;
    do {
        d = (d + 1) / 2;
        f = 0;
        for (int i = 0; i < l - d; i++) {
            if (a[i] > a[i + d]) {
                std::swap(a[i + d], a[i]);
                f = 1;
            }
        }
        if (d == 1) {
            l--;
        }
    } while (!((d == 1) && (f == 0)));
    return a;
}

std::vector<int> merge(std::vector<int> a, std::vector<int> b) {
    int size1 = a.size();
    int size2 = b.size();
    std::vector<int> c(size1 + size2);
    int i, j, k;
    i = j = k = 0;
    while (i < size1 && j < size2) {
        if (a[i] <= b[j]) {
            c[k] = a[i];
            i++;
        } else {
            c[k] = b[j];
            j++;
        }
        k++;
    }
    while (i < size1) {
        c[k] = a[i];
        i++;
        k++;
    }

    while (j < size2) {
        c[k] = b[j];
        j++;
        k++;
    }
    return c;
}

std::vector<int> genvector(int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    std::vector<int> vec(size);
    for (int i = 0; i < size; i++) {
        vec[i] = gen() % 10000;
    }
    return vec;
}


std::vector<int> parallelsortshell(std::vector<int> vec, int n) {
    int procsize, rank;
    double t1, t2;
    MPI_Comm_size(MPI_COMM_WORLD, &procsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    t1 = MPI_Wtime();

    int delta = n / procsize;
    int k = n % procsize;

    if (rank == 0) {
        for (int proc = 1; proc < procsize; proc++) {
            MPI_Send(vec.data() + proc * delta + k, delta, MPI_INT, proc, 0, MPI_COMM_WORLD);
        }
    }

    std::vector<int> localVec(delta);

    if (rank == 0) {
        localVec = std::vector<int>(delta + k);
    } else {
        localVec = std::vector<int>(delta);
    }

    if (rank == 0) {
        localVec = std::vector<int>(vec.begin(), vec.begin() + delta + k);
    } else {
        MPI_Status status;
        MPI_Recv(localVec.data(), delta, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }

    int size = localVec.size();
    localVec = sortshell(localVec, size);

    int p = procsize;
    int m = 1;

    while (p > 1) {
        p = p / 2 + p % 2;
        if ((rank - m) % (2 * m) == 0) {
            MPI_Send(&size, 1, MPI_INT, rank - m, 0, MPI_COMM_WORLD);
            MPI_Send(localVec.data(), size, MPI_INT, rank - m, 0, MPI_COMM_WORLD);
        }

        if ((rank % (2 * m) == 0) && ((procsize - rank) > m)) {
            MPI_Status status;
            int localSize;
            MPI_Recv(&localSize, 1, MPI_INT, rank + m, 0, MPI_COMM_WORLD, &status);
            std::vector <int> localVec2(localSize);
            MPI_Recv(localVec2.data(), localSize, MPI_INT, rank + m, 0, MPI_COMM_WORLD, &status);
            localVec = merge(localVec, localVec2);
            size = size + localSize;
        }
        m = 2 * m;
    }
    t2 = MPI_Wtime();
    if (rank == 0) {
        std::cout << "parallel time: " << t2 - t1 << std::endl;
    }
    return localVec;
}

