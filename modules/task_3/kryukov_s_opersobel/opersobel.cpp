// Copyright 2020 Kryukov Sergey
#include <mpi.h>
#include <random>
#include <ctime>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include "../../../modules/task_3/kryukov_s_opersobel/opersobel.h"

std::vector<int> sobel_for_oneproc(std::vector<int> img,  int row, int col) {
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1 };
    std::vector<int> kernelY = { -1, -2, 1, 0,  0,  0, 1,  2,  1 };
    std::vector<int> image(col * row);
    for (int x = 0; x < row; x++) {
        for (int y = 0; y < col; y++) {
            int Ggor = 0, Gvert = 0;
            int Gresult = 0;

            if (x == 0 || x == row) {
                Gresult = 0;
            } else if (y == 0 || y == col) {
                Gresult = 0;
            } else {
                for (int i = -1; i < 2; i++) {
                    for (int j = -1; j < 2; j++) {
                        int idx = (i + 1) * 3 + j + 1;
                        int X = correctRange(x + j, row - 1, 0);
                        int Y = correctRange(y + i, col - 1, 0);
                        Ggor += img[X * col + Y] * kernelX[idx];
                        Gvert += img[X * col + Y] * kernelY[idx];
                    }
                }
            }
            Gresult = sqrt(pow(Ggor, 2) + pow(Gvert, 2));
            Gresult = correctRange(Gresult, 255, 0);
            image[y + x * col] = Gresult;
        }
    }
    return image;
}

int use_sobel_kernel(std::vector<int> img, int x, int y, int row, int col) {
    std::vector<int> kernelX = { -1, 0, 1, -2, 0, 2, -1, 0, 1};
    std::vector<int> kernelY = { -1, -2, 1, 0,  0,  0, 1,  2,  1};
    int Ggor = 0, Gvert = 0;
    int Gresult = 0;
    if (x == 0 || x == row) {
        Gresult = 0;
    } else if (y == 0 || y == col) {
        Gresult = 0;
    } else {
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                int idx = (i + 1) * 3 + j + 1;
                int X = correctRange(x + j, row - 1, 0);
                int Y = correctRange(y + i, col - 1, 0);
                Ggor += img[X * col + Y] * kernelX[idx];
                Gvert += img[X * col + Y] * kernelY[idx];
            }
        }
    }
    Gresult = sqrt(pow(Ggor, 2) + pow(Gvert, 2));
    Gresult = correctRange(Gresult, 255, 0);
    return Gresult;
}

std::vector<int> paral_sobel_oper(std::vector<int> img, int row, int col) {
    int procsize, procrank;
    MPI_Comm_size(MPI_COMM_WORLD, &procsize);
    MPI_Comm_rank(MPI_COMM_WORLD, &procrank);
    if ((row < procsize) || (procsize == 1)) {
        return sobel_for_oneproc(img, row, col);
    }
    int* buf = new int[row];
    int* sendcountScat = new int[procsize];
    int delta = row / procsize;
    int remainder = row % procsize;
    int* displsScat = new int[procsize];

    for (int i = 0; i < row; i++) {
        buf[i] = i;
    }

    for (int i = 0; i < procsize; i++) {
        sendcountScat[i] = delta + !!remainder;
        if (remainder) {
            --remainder;
        }
    }
    int k = 0;
    for (int i = 0; i < procsize; i++) {
        displsScat[i] = k;
        k += sendcountScat[i];
    }
    std::vector<int> recvbuf(sendcountScat[procrank]);
    MPI_Scatterv(&buf[0], sendcountScat, displsScat, MPI_INT, &recvbuf[0], sendcountScat[procrank],
        MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> res_local(col * recvbuf.size());
    for (int i = 0; i < sendcountScat[procrank]; i++) {
        for (int j = 0; j < col; j++) {
            res_local[i * col + j] = use_sobel_kernel(img, recvbuf[i], j, row, col);
        }
    }
    std::vector<int> res_global(row * col);
    int* sendcountGat = new int[procsize];
    int* displsGat = new int[procsize];
    for (int i = 0; i < procsize; i++) {
        sendcountGat[i] = col * recvbuf.size();
    }
    k = 0;
    for (int i = 0; i < procsize; i++) {
        displsGat[i] = k * col;
        k += sendcountScat[i];
    }
    MPI_Gatherv(res_local.data(), sendcountGat[procrank], MPI_INT, res_global.data(),
        sendcountGat, displsGat, MPI_INT, 0, MPI_COMM_WORLD);
    return res_global;
}


int correctRange(int value , int max, int min) {
    if (value > max) return max;
    else if (value < min) return min;
    return  value;
}
