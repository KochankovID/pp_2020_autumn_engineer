  // Copyright 2020 Ivanov Yaroslav
#include <mpi.h>
#include <random>
#include <algorithm>
#include <ctime>
#include <vector>
#include <iostream>
#include "../../../modules/task_3/ivanov_y_linefilter_block/linefilter_block.h"

void Show(std::vector<int> lol, int height, int width) {
    for (int i = 0; i < height * width; i++) {
        std::cout << lol[i] << " ";
        if ((i + 1) % width == 0) {
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

std::vector<int> CreateRandomImg(const int width, const int height) {
    std::mt19937 gen;
    int pixes = height * width;
    static int change = 0;
    gen.seed(static_cast<int>(time(0)) + (change += 5));
    std::vector<int> Arr(pixes);
    for (int i = 0; i < width * height; i++) {
        Arr[i] = gen() % 255;
    }
    return Arr;
}

int createDiameter(int radius) {
    return radius * 2 + 1;
}

std::vector<float> createKernel(int radius, float sigma) {
    int diam = createDiameter(radius);
    float normalization = 0;
    std::vector<float> kernel(diam * diam);
    for (int i = -radius; i <= radius; i++) {
        for (int j = -radius; j <= radius; j++) {
            int index = (j + radius) + ((i + radius) * diam);
            kernel[index] = exp(-(pow(i, 2) + pow(j, 2)) / (pow(sigma, 2)));
            normalization += kernel[index];
        }
    }
    for (int i = 0; i < pow(diam, 2); i++) {
        kernel[i] = kernel[i] / normalization;
    }
    return kernel;
}

int checkBoundaries(int data, int left, int right) {
    if (data < left) {
        data = left;
    } else {
        if (data > right) {
            data = right;
        }
    }
    return data;
}

int LocalFilter(const std::vector<int> &Arr, int height, int width, int radius, float sigma,
    const std::vector<float> &Kernel, int row, int cols) {
    // std::vector<int> Local((height * width) - (begin * width));
    int Local = 0;
    int diam = createDiameter(radius);
    float sum = 0;
    for (int i = -radius; i < radius; i++) {
        for (int j = -radius; j < radius; j++) {
            int index = (j + radius) + ((i + radius) * diam);
            int x = checkBoundaries(row + j, 0, height - 1);
            int y = checkBoundaries(cols + i, 0, width - 1);
            float color = Arr[x * width + y];
            sum += color * Kernel[index];
        }
    }
    return Local = checkBoundaries(sum, 0, 255);
}

std::vector<int> ParallelFilterDeep(const std::vector<int>& Arr, int height, int width, int radius, float sigma,
    const std::vector<float>& Kernel, int begin, int add) {
    std::vector<int> local_result_mas(add * width);
    for (int i = 0; i < add; i++) {
        for (int j = 0; j < width; j++) {
            local_result_mas[i * width + j] = LocalFilter(Arr, height, width, radius, sigma, Kernel, i + begin, j);
        }
    }
    return local_result_mas;
}

std::vector<int> ParallelFilter(const std::vector<int> &Arr, int height, int width, int radius, float sigma,
 const std::vector<float>& Kernel) {
    int process_number, process_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &process_number);
    MPI_Comm_rank(MPI_COMM_WORLD, &process_rank);

    int del = height / process_number;
    int tmp = height % process_number;
    int pixes = height * width;
    int* sendbuf = new int[height];
    int* sendcounts = new int[process_number];
    int* displs = new int[process_number];
    for (int i = 0; i < height; i++) {
        sendbuf[i] = i;
    }
    for (int i = 0; i < process_number; i++) {
        if (tmp > 0) {
            sendcounts[i] = del + 1;
            tmp = tmp - 1;
        } else {
            sendcounts[i] = del;
        }
    }
    int justFlag = 0;
    for (int i = 0; i < process_number; i++) {
        displs[i] = justFlag;
        justFlag += sendcounts[i];
    }
    std::vector<int> recvbuf(sendcounts[process_rank]);
    MPI_Scatterv(&sendbuf[0], sendcounts, displs, MPI_INT, &recvbuf[0],
        sendcounts[process_rank], MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<int>local_result_mas(recvbuf.size() * width);
    int b = 0;
    for (int i = 0; i < process_rank; i++) {
        b += sendcounts[i];
    }
    local_result_mas = ParallelFilterDeep(Arr, height, width, radius, sigma,
 Kernel, b, sendcounts[process_rank]);
    std::vector<int> global_result(pixes);
    int* sendcounts_global = new int[process_number];
    int* displs_global = new int[process_number];
    for (int i = 0; i < process_number; i++) {
        sendcounts_global[i] = static_cast<int>(width * recvbuf.size());
    }
    int justFlag1 = 0;
    for (int i = 0; i < process_number; i++) {
        displs_global[i] = justFlag1 * width;
        justFlag1 += sendcounts[i];
    }
    MPI_Gatherv(&local_result_mas[0], sendcounts_global[process_rank],
        MPI_INT, &global_result[0], sendcounts_global, displs_global, MPI_INT,
 0, MPI_COMM_WORLD);
    return global_result;
}
