// Copyright 2020 Zhafyarov Oleg
#ifndef MODULES_TASK_3_ZHAFYAROV_O_CRS_MATRIX_CRS_MATRIX_H_
#define MODULES_TASK_3_ZHAFYAROV_O_CRS_MATRIX_CRS_MATRIX_H_
#include <vector>
struct MatrixCRS {
  int Size;
  int No_empty;
  std::vector<double> Value;
  std::vector<int> Col;
  std::vector<int> Pointer;
};

void InitializeMatrix(int Size, int No_empty, MatrixCRS* Matrix);

double Generate(int min, int max);

void RandomMatrixCRS(int Size, int No_empty, MatrixCRS* Matrix);

MatrixCRS Transpose(int Size, int Nonzero, int Nonzero_in_row, MatrixCRS* Matrix);

void Multiplication(MatrixCRS* A, MatrixCRS* B, MatrixCRS* Result);

void Split(int process_rank, int Size, MatrixCRS* mtx1, MatrixCRS* mtx2, int No_empty_in_row);

void RowIndexProc(int N, MatrixCRS* mtx, int cntInRow);

void ParallelMultiplication(int process_rank, int Size, MatrixCRS* A, MatrixCRS* B, MatrixCRS* Result);

#endif  // MODULES_TASK_3_ZHAFYAROV_O_CRS_MATRIX_CRS_MATRIX_H_
