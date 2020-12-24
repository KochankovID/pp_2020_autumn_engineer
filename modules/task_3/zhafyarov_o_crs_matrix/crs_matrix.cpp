  // Copyright 2020 Zhafyarov Oleg
#include <mpi.h>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <ctime>
#include "../../../modules/task_3/zhafyarov_o_crs_matrix/crs_matrix.h"

int process_number;

void InitializeMatrix(int Size, int No_empty, MatrixCRS* Matrix) {
  Matrix->Size = Size;
  Matrix->No_empty = No_empty;
  Matrix->Value = std::vector<double>(static_cast<size_t>(No_empty));
  Matrix->Col = std::vector<int>(static_cast<size_t>(No_empty));
  Matrix->Pointer = std::vector<int>(static_cast<size_t>(Size + 1));
}

double Generate(int min, int max) {
  static int count = 0;
  std::mt19937 generator(count++);
  std::uniform_int_distribution<> distribution{ min, max };
  double res = distribution(generator) / 4.;
  return res;
}

void RandomMatrixCRS(int Size, int No_empty, MatrixCRS* Matrix) {
  int f, tmp;
  for (int i = 0; i < Size; i++) {
    for (int j = 0; j < No_empty; j++) {
      do {
        Matrix->Col[i * No_empty + j] = Generate(1, Size * 4);
        f = 0;
        for (int k = 0; k < j; k++) {
          if (Matrix->Col[i * No_empty + j] == Matrix->Col[i * No_empty + k]) {
            f = 1;
          }
        }
      } while (f == 1);
    }
    for (int j = 0; j < No_empty - 1; j++) {
      for (int k = 0; k < No_empty - 1; k++) {
        if (Matrix->Col[i * No_empty + k] > Matrix->Col[i * No_empty + k + 1]) {
          tmp = Matrix->Col[i * No_empty + k];
          Matrix->Col[i * No_empty + k] = Matrix->Col[i * No_empty + k + 1];
          Matrix->Col[i * No_empty + k + 1] = tmp;
        }
      }
    }
  }
  for (int i = 0; i < No_empty * Size; i++) {
    Matrix->Value[i] = Generate(5, 25);
  }
  int c = 0;
  for (int i = 0; i <= Size; i++) {
    Matrix->Pointer[i] = c;
    c += No_empty;
  }
}

MatrixCRS Transpose(int Size, int Nonzero, int Nonzero_in_row, MatrixCRS* Matrix) {
  MatrixCRS Matrix_T;
  InitializeMatrix(Size, Nonzero, &Matrix_T);

  for (int i = 0; i < Matrix->No_empty; i++) {
    Matrix_T.Pointer[Matrix->Col[i] + 1]++;
  }

  int Index_tmp = 0;
  for (int i = 1; i <= Matrix->Size; i++) {
    int tmp = Matrix_T.Pointer[i];
    Matrix_T.Pointer[i] = Index_tmp;
    Index_tmp = Index_tmp + tmp;
  }
  for (int i = 0; i < Matrix->Size; i++) {
    int j1 = Matrix->Pointer[i];
    int j2 = Matrix->Pointer[i + 1];
    int Col = i;
    for (int j = j1; j < j2; j++) {
      double V = Matrix->Value[j];
      int Row_index = Matrix->Col[j];
      int IIndex = Matrix_T.Pointer[Row_index + 1];
      Matrix_T.Value[IIndex] = V;
      Matrix_T.Col[IIndex] = Col;
      Matrix_T.Pointer[Row_index + 1]++;
    }
  }
  return Matrix_T;
}

void Multiplication(MatrixCRS* A, MatrixCRS* B, MatrixCRS* Result) {
  int N = A->Size; int N2 = B->Size;
  double sum;

  std::vector<double> values;
  std::vector<int> col;
  std::vector<int> pointer;

  int rowNZ; pointer.push_back(0);
  for (int i = 0; i < N; i++) {
    rowNZ = 0;
    for (int j = 0; j < N2; j++) {
      sum = 0.0;
      for (int k = A->Pointer[i]; k < A->Pointer[i + 1]; k++) {
        for (int l = B->Pointer[j]; l < B->Pointer[j + 1]; l++) {
          if (A->Col[k] == B->Col[l]) {
            if (A->Col[k] == B->Col[l]) {
              sum += A->Value[k] * B->Value[l];
            }
            break;
          }
        }
      }

      if (fabs(sum) > 0) {
        col.push_back(j);
        values.push_back(sum);
        rowNZ++;
      }
    }
    pointer.push_back(rowNZ + pointer[i]);
  }
  InitializeMatrix(N, col.size(), Result);

  for (int j = 0; j < static_cast<int>(col.size()); j++) {
    Result->Col[j] = col[j];
    Result->Value[j] = values[j];
  }
  for (int i = 0; i <= N; i++) {
    Result->Pointer[i] = pointer[i];
  }
}

void Split(int process_rank, int Size, MatrixCRS* mtx1, MatrixCRS* mtx2, int No_empty_in_row) {
  int Left_send_rows = Size;
  int Send_number_row = (Size / process_number);
  int* Send_size_for_process = new int[process_number];
  int* Recv_index_for_process = new int[process_number];

  Send_size_for_process[0] = Send_number_row * No_empty_in_row;
  Recv_index_for_process[0] = 0;

  for (int i = 1; i < process_number; i++) {
    Left_send_rows -= Send_number_row;
    Send_number_row = Left_send_rows / (process_number - i);
    Send_size_for_process[i] = Send_number_row * No_empty_in_row;
    Recv_index_for_process[i] = Recv_index_for_process[i - 1] + Send_size_for_process[i - 1];
  }

  MPI_Scatterv(&mtx1->Value.at(0), Send_size_for_process, Recv_index_for_process, MPI_DOUBLE,
    &mtx2->Value.at(0), Send_size_for_process[process_rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

  MPI_Scatterv(&mtx1->Col.at(0), Send_size_for_process, Recv_index_for_process, MPI_INT,
    &mtx2->Col.at(0), Send_size_for_process[process_rank], MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(&mtx1->Pointer.at(0), Send_size_for_process, Recv_index_for_process, MPI_INT,
    &mtx2->Pointer.at(0), Send_size_for_process[process_rank], MPI_INT, 0, MPI_COMM_WORLD);
}

void RowIndexProc(int N, MatrixCRS* mtx, int cntInRow) {
  mtx[0].Pointer[0] = 0;
  for (int i = 1; i < N + 1; i++)
    mtx[0].Pointer[i] = mtx[0].Pointer[i - 1] + cntInRow;
}

void ParallelMultiplication(int process_rank, int Size, MatrixCRS* A, MatrixCRS* B, MatrixCRS* Result) {
  MPI_Comm_size(MPI_COMM_WORLD, &process_number);
  if (process_number >= Size) {
    if (process_rank == 0) {
      Multiplication(A, B, Result);
      return;
    } else {
      return;
    }
  }
  int Nonzero_in_row = 1;
  int All_No_empty = Nonzero_in_row * Size;

  MatrixCRS Tmp;
  InitializeMatrix(Size, Size, Result);
  int Left_rows = Size;

  int* Recv_size_from_process = new int[process_number];
  int* Recv_index_from_process = new int[process_number];
  Recv_index_from_process[0] = 0;
  Recv_size_from_process[0] = Size / process_number;
  for (int i = 1; i < process_number; i++) {
    Left_rows -= Recv_size_from_process[i - 1];
    Recv_size_from_process[i] = Left_rows / (process_number - i);
    Recv_index_from_process[i] = Recv_index_from_process[i - 1] + Recv_size_from_process[i - 1];
  }

  MPI_Bcast(&B->Pointer.at(0), Size + 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B->Value.at(0), All_No_empty, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&B->Col.at(0), All_No_empty, MPI_INT, 0, MPI_COMM_WORLD);
  InitializeMatrix(Recv_size_from_process[process_rank], Recv_size_from_process[process_rank] * Nonzero_in_row, &Tmp);
  Split(process_rank, Size, A, &Tmp, Nonzero_in_row);
  RowIndexProc(Recv_size_from_process[process_rank], &Tmp, Nonzero_in_row);

  MatrixCRS Tmp_result;
  InitializeMatrix(Size, Size, &Tmp_result);
  Multiplication(&Tmp, B, &Tmp_result);

  MPI_Gatherv(&Tmp_result.Value.at(0), Tmp_result.Size, MPI_DOUBLE,
    &Result->Value.at(0), Recv_size_from_process, Recv_index_from_process, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Gatherv(&Tmp_result.Col.at(0), Tmp_result.Size, MPI_INT,
    &Result->Col.at(0), Recv_size_from_process, Recv_index_from_process, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Gatherv(&Tmp_result.Pointer.at(0), Tmp_result.Size, MPI_INT,
    &Result->Pointer.at(0), Recv_size_from_process, Recv_index_from_process, MPI_INT, 0, MPI_COMM_WORLD);
}
