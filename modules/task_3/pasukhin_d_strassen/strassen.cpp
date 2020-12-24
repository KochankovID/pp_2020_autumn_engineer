// Copyright 2020 Pasukhin Dmitry

#include "../../../modules/task_3/pasukhin_d_strassen/strassen.h"

#include <mpi.h>

#include <iostream>
#include <random>
#include <vector>

int comm_rank = 1;
int comm_size = 1;
int rows = 1;
int columns = 1;

int checkProc(const int size) {
  int pow = 1;
  for (int i = 0; i <= 10; ++i) {
    if (size == pow) return i;
    pow *= 4;
  }
  return -1;
}

void Form_Matrix(const std::vector<double>& C11, const std::vector<double>& C12,
                 const std::vector<double>& C21, const std::vector<double>& C22,
                 const int size, double* C) {
  int i4 = -size / 2, i2 = 0, i3 = i4;

  for (int i = 0; i < size; i++) {
    int j2 = 0;
    int j3 = 0;
    for (int j = 0; j < size; j++) {
      if (j < (size / 2) && (i < size / 2))
        C[i * size + j] = C11[i * (size / 2) + j];

      if (j >= size / 2 && i < size / 2) {
        C[i * size + j] = C12[i2 * (size / 2) + j2];
        j2++;
      }

      if (j < size / 2 && i >= size / 2)
        C[i * size + j] = C21[i3 * (size / 2) + j];

      if (i >= size / 2 && j >= size / 2) {
        C[i * size + j] = C22[i4 * (size / 2) + j3];
        j3++;
      }
    }
    i2++;
    i3++;
    i4++;
  }
}

void Sum(const std::vector<double>& A, const std::vector<double>& B,
         std::vector<double>* res, const int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      res->at(i * size + j) = A[i * size + j] + B[i * size + j];
    }
  }
}

void Sub(const std::vector<double>& A, const std::vector<double>& B,
         std::vector<double>* res, const int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      res->at(i * size + j) = A[i * size + j] - B[i * size + j];
    }
  }
}

void Serial_Strassen(const std::vector<double>& A, const std::vector<double>& B,
                     std::vector<double>* C, const int size) {
  Serial_Strassen(&A.at(0), &B.at(0), &C->at(0), size);
}

void CutSybMatr(const double* origin, std::vector<double>* res,
                const int orig_colms, const int new_columns) {
  for (int i = 0; i < new_columns; i++) {
    for (int j = 0; j < new_columns; j++)
      res->at(i * new_columns + j) = origin[i * orig_colms + j];
  }
}

void Serial_Strassen(const double* A, const double* B, double* C,
                     const int size) {
  const int H = size / 2;
  if (size * size <= 16) {
    Serial_Multiply(A, B, C, size);
    return;
  }
  std::vector<double> A11(H * H, 0), A12(H * H, 0), A21(H * H, 0),
      A22(H * H, 0), B11(H * H, 0), B12(H * H, 0), B21(H * H, 0), B22(H * H, 0);
  std::vector<double> P1(H * H, 0), P2(H * H, 0), P3(H * H, 0), P4(H * H, 0),
      P5(H * H, 0), P6(H * H, 0), P7(H * H, 0);

  std::vector<double> T1(H * H, 0);
  std::vector<double> T2(H * H, 0);
  std::vector<double> T3(H * H, 0);
  std::vector<double> T4(H * H, 0);
  std::vector<double> T5(H * H, 0);
  std::vector<double> T6(H * H, 0);
  std::vector<double> T7(H * H, 0);
  std::vector<double> T8(H * H, 0);
  std::vector<double> T9(H * H, 0);
  std::vector<double> T10(H * H, 0);
  std::vector<double> T11(H * H, 0);
  std::vector<double> T12(H * H, 0);
  std::vector<double> T13(H * H, 0);
  std::vector<double> T14(H * H, 0);

  CutSybMatr(A, &A11, size, H);
  CutSybMatr(B, &B11, size, H);

  CutSybMatr(A + size / 2, &A12, size, H);
  CutSybMatr(B + size / 2, &B12, size, H);

  CutSybMatr(A + size * size / 2, &A21, size, H);
  CutSybMatr(B + size * size / 2, &B21, size, H);

  CutSybMatr(A + size * size / 2 + size / 2, &A22, size, H);
  CutSybMatr(B + size * size / 2 + size / 2, &B22, size, H);

  Sum(A11, A22, &T1, H);
  Sum(B11, B22, &T2, H);
  Sum(A21, A22, &T3, H);
  T4 = B11;
  T5 = A11;
  Sub(B12, B22, &T6, H);
  T7 = A22;
  Sub(B21, B11, &T8, H);
  Sum(A11, A12, &T9, H);
  T10 = B22;
  Sub(A21, A11, &T11, H);
  Sum(B11, B12, &T12, H);
  Sub(A12, A22, &T13, H);
  Sum(B21, B22, &T14, H);

  Serial_Strassen(T1, T2, &P1, H);
  Serial_Strassen(T3, T4, &P2, H);
  Serial_Strassen(T5, T6, &P3, H);
  Serial_Strassen(T7, T8, &P4, H);
  Serial_Strassen(T9, T10, &P5, H);
  Serial_Strassen(T11, T12, &P6, H);
  Serial_Strassen(T13, T14, &P7, H);

  std::vector<double> C11(H * H, 0);
  std::vector<double> C12(H * H, 0);
  std::vector<double> C21(H * H, 0);
  std::vector<double> C22(H * H, 0);

  for (int i = 0; i < H * H; i++) {
    C11.at(i) = P1.at(i) + P4.at(i) - P5.at(i) + P7.at(i);
    C12.at(i) = P3.at(i) + P5.at(i);
    C21.at(i) = P2.at(i) + P4.at(i);
    C22.at(i) = P1.at(i) + P3.at(i) - P2.at(i) + P6.at(i);
  }

  Form_Matrix(C11, C12, C21, C22, size, C);
}

void GenerateMatrix(double* matrix, const int size, const int seed) {
  std::mt19937 generator(seed);
  std::uniform_int_distribution<> distribution{5, 50};
  for (int i = 0; i < size; ++i) {
    matrix[i] = distribution(generator) / 4.;
  }
}

void Serial_Multiply(const double* A, const double* B, double* C,
                     const int size) {
  for (int i = 0; i < size; i++) {
    for (int j = 0; j < size; j++) {
      double sum = 0;
      for (int k = 0; k < size; k++) {
        sum += A[i * size + k] * B[k * size + j];
      }
      C[i * size + j] = sum;
    }
  }
}

void Spetial_Strassen(const std::vector<double>& A, std::vector<double>* B,
                      std::vector<double>* C, const int root, const int size) {
  if (rows % comm_size != 0) {
    Serial_Strassen(A, *B, C, rows);
    return;
  }

  std::vector<double> result(size, 0);

  const int from = (comm_rank * rows) / comm_size;
  const int to = ((comm_rank + 1) * rows) / comm_size;

  MPI_Bcast(&B->at(0), size, MPI_DOUBLE, root, MPI_COMM_WORLD);
  const int elements_per_proc = size / comm_size;

  std::vector<double> sub(elements_per_proc, 0);
  MPI_Scatter(&A.at(0), elements_per_proc, MPI_DOUBLE, &sub.at(0),
              elements_per_proc, MPI_DOUBLE, root, MPI_COMM_WORLD);
  int ci = 0;
  for (int i = from; i < to; i++) {
    for (int j = 0; j < columns; j++) {
      double sum = 0;
      for (int k = 0; k < rows; k++) {
        sum += sub[ci * rows + k] * B->at(k * rows + j);
      }
      result[ci * rows + j] = sum;
    }
    ci++;
  }
  MPI_Gather(&result.at(0), size / comm_size, MPI_DOUBLE, &C->at(0),
             size / comm_size, MPI_DOUBLE, root, MPI_COMM_WORLD);
}

void Strassen_Multiplication(const double* A, const double* B, double* C,
                             const int size) {
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

  int thread_size = static_cast<int>(std::sqrt(comm_size));
  rows = (size / 2);
  columns = rows;
  int pow = checkProc(comm_size);
  bool isOk = pow != -1 ? true : false;

  if (!isOk || size * size < comm_size || rows % 2 != 0 ||
      (size * size) % comm_size != 0) {
    Serial_Multiply(A, B, C, size);
    return;
  } else if (comm_size == 1) {
    Serial_Strassen(A, B, C, size);
    return;
  }
  thread_size = 2;

  rows = (size / 2);
  columns = rows;
  const int no_of_elements = rows * columns;

  std::vector<double> local_A(no_of_elements);
  std::vector<double> local_B(no_of_elements);
  std::vector<double> local_C(no_of_elements);

  std::vector<double> T1(no_of_elements);
  std::vector<double> T2(no_of_elements);
  std::vector<double> T3(no_of_elements);
  std::vector<double> T4(no_of_elements);
  std::vector<double> T5(no_of_elements);
  std::vector<double> T6(no_of_elements);
  std::vector<double> T7(no_of_elements);
  std::vector<double> T8(no_of_elements);
  std::vector<double> T9(no_of_elements);
  std::vector<double> T10(no_of_elements);
  std::vector<double> T11(no_of_elements);
  std::vector<double> T12(no_of_elements);
  std::vector<double> T13(no_of_elements);
  std::vector<double> T14(no_of_elements);
  std::vector<double> P1(no_of_elements);
  std::vector<double> P2(no_of_elements);
  std::vector<double> P3(no_of_elements);
  std::vector<double> P4(no_of_elements);
  std::vector<double> P5(no_of_elements);
  std::vector<double> P6(no_of_elements);
  std::vector<double> P7(no_of_elements);

  for (int i = 0; i < thread_size; i += 2) {
    for (int j = 0; j < thread_size; j += 2) {
      if (comm_rank != (thread_size)*i + j) continue;

      std::vector<double> local_A22(no_of_elements);
      std::vector<double> local_B22(no_of_elements);
      std::vector<double> local_A12(no_of_elements);
      std::vector<double> local_B21(no_of_elements);

      CutSybMatr(A, &local_A, size, rows);
      CutSybMatr(B, &local_B, size, rows);

      CutSybMatr(A + size / 2, &local_A12, size, rows);

      CutSybMatr(B + size * size / 2, &local_B21, size, rows);

      CutSybMatr(A + size * size / 2 + size / 2, &local_A22, size, rows);
      CutSybMatr(B + size * size / 2 + size / 2, &local_B22, size, rows);

      for (int i = 0; i < no_of_elements; i++) {
        T1.at(i) = local_A.at(i) + local_A22.at(i);
        T2.at(i) = local_B.at(i) + local_B22.at(i);
        T3.at(i) = local_A12.at(i) - local_A22.at(i);
        T4.at(i) = local_B21.at(i) + local_B22.at(i);
      }
    }
  }

  for (int i = 0; i < thread_size; i += 2) {
    for (int j = 1; j < thread_size; j += 2) {
      if (comm_rank != (thread_size)*i + j) continue;

      std::vector<double> local_A11(no_of_elements);
      std::vector<double> local_B22(no_of_elements);

      CutSybMatr(A, &local_A11, size, rows);

      CutSybMatr(A + size / 2, &local_A, size, rows);
      CutSybMatr(B + size / 2, &local_B, size, rows);

      CutSybMatr(B + size * size / 2 + size / 2, &local_B22, size, rows);

      for (int i = 0; i < no_of_elements; i++) {
        T5.at(i) = local_B.at(i) - local_B22.at(i);
        T6.at(i) = local_A11.at(i) + local_A.at(i);
        T7.at(i) = local_B22.at(i);
        T14.at(i) = local_A11.at(i);
      }
    }
  }

  for (int i = 1; i < thread_size; i += 2) {
    for (int j = 0; j < thread_size; j += 2) {
      if (comm_rank != (thread_size)*i + j) continue;

      std::vector<double> local_A22(no_of_elements);
      std::vector<double> local_B11(no_of_elements);

      CutSybMatr(B, &local_B11, size, rows);

      CutSybMatr(A + size * size / 2, &local_A, size, rows);
      CutSybMatr(B + size * size / 2, &local_B, size, rows);

      CutSybMatr(A + size * size / 2 + size / 2, &local_A22, size, rows);

      for (int i = 0; i < no_of_elements; i++) {
        T8.at(i) = local_A.at(i) + local_A22.at(i);
        T9.at(i) = local_B11.at(i);
        T10.at(i) = local_A22.at(i);
        T11.at(i) = local_B.at(i) - local_B11.at(i);
      }
    }
  }

  for (int i = 1; i < thread_size; i += 2) {
    for (int j = 1; j < thread_size; j += 2) {
      if (comm_rank != (thread_size)*i + j) continue;

      std::vector<double> local_A21(no_of_elements);
      std::vector<double> local_A11(no_of_elements);
      std::vector<double> local_B12(no_of_elements);
      std::vector<double> local_B11(no_of_elements);

      CutSybMatr(A, &local_A11, size, rows);
      CutSybMatr(B, &local_B11, size, rows);

      CutSybMatr(B + size / 2, &local_B12, size, rows);

      CutSybMatr(A + size * size / 2, &local_A21, size, rows);

      CutSybMatr(A + size * size / 2 + size / 2, &local_A, size, rows);
      CutSybMatr(B + size * size / 2 + size / 2, &local_B, size, rows);

      for (int i = 0; i < no_of_elements; i++) {
        T12.at(i) = local_A21.at(i) - local_A11.at(i);
        T13.at(i) = local_B11.at(i) + local_B12.at(i);
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  Spetial_Strassen(T1, &T2, &P1, 0, no_of_elements);
  Spetial_Strassen(T3, &T4, &P7, 0, no_of_elements);
  Spetial_Strassen(T14, &T5, &P3, 1, no_of_elements);
  Spetial_Strassen(T6, &T7, &P5, 1, no_of_elements);
  Spetial_Strassen(T8, &T9, &P2, 2, no_of_elements);
  Spetial_Strassen(T10, &T11, &P4, 2, no_of_elements);
  Spetial_Strassen(T12, &T13, &P6, 3, no_of_elements);

  for (int i = 0; i < thread_size; i += 2) {
    for (int j = 1; j < thread_size; j += 2) {
      if (comm_rank != (thread_size)*i + j) continue;

      MPI_Request req;
      MPI_Isend(&P3.at(0), no_of_elements, MPI_DOUBLE, comm_rank - 1, 100,
                MPI_COMM_WORLD, &req);
      MPI_Isend(&P5.at(0), no_of_elements, MPI_DOUBLE, comm_rank - 1, 200,
                MPI_COMM_WORLD, &req);
    }
  }

  for (int i = 1; i < thread_size; i += 2) {
    for (int j = 0; j < thread_size; j += 2) {
      if (comm_rank != (thread_size)*i + j) continue;

      MPI_Request req;
      MPI_Isend(&P2.at(0), no_of_elements, MPI_DOUBLE, comm_rank - thread_size,
                300, MPI_COMM_WORLD, &req);
      MPI_Isend(&P4.at(0), no_of_elements, MPI_DOUBLE, comm_rank - thread_size,
                400, MPI_COMM_WORLD, &req);
    }
  }

  for (int i = 1; i < thread_size; i += 2) {
    for (int j = 1; j < thread_size; j += 2) {
      if (comm_rank != (thread_size)*i + j) continue;

      MPI_Request req;
      MPI_Isend(&P6.at(0), no_of_elements, MPI_DOUBLE,
                comm_rank - thread_size - 1, 500, MPI_COMM_WORLD, &req);
    }
  }

  std::vector<double> RES(no_of_elements * 4, 0);
  for (int i = 0; i < thread_size; i += 2) {
    for (int j = 0; j < thread_size; j += 2) {
      if (comm_rank != (thread_size)*i + j) continue;
      MPI_Status stat;
      MPI_Recv(&P3.at(0), no_of_elements, MPI_DOUBLE, comm_rank + 1, 100,
               MPI_COMM_WORLD, &stat);
      MPI_Recv(&P5.at(0), no_of_elements, MPI_DOUBLE, comm_rank + 1, 200,
               MPI_COMM_WORLD, &stat);
      MPI_Recv(&P2.at(0), no_of_elements, MPI_DOUBLE, comm_rank + thread_size,
               300, MPI_COMM_WORLD, &stat);
      MPI_Recv(&P4.at(0), no_of_elements, MPI_DOUBLE, comm_rank + thread_size,
               400, MPI_COMM_WORLD, &stat);
      MPI_Recv(&P6.at(0), no_of_elements, MPI_DOUBLE,
               comm_rank + thread_size + 1, 500, MPI_COMM_WORLD, &stat);
      std::vector<double> C11(no_of_elements);
      std::vector<double> C12(no_of_elements);
      std::vector<double> C21(no_of_elements);
      std::vector<double> C22(no_of_elements);

      for (int i = 0; i < no_of_elements; i++) {
        C11.at(i) = P1.at(i) + P4.at(i) - P5.at(i) + P7.at(i);
        C12.at(i) = P3.at(i) + P5.at(i);
        C21.at(i) = P2.at(i) + P4.at(i);
        C22.at(i) = P1.at(i) + P3.at(i) - P2.at(i) + P6.at(i);
      }
      Form_Matrix(C11, C12, C21, C22, rows * 2, &RES.at(0));

      MPI_Request req;
      MPI_Isend(&RES.at(0), no_of_elements * 4, MPI_DOUBLE, 0, 1000,
                MPI_COMM_WORLD, &req);
    }
  }

  if (comm_rank == 0) {
    for (int i = 0; i < thread_size; i += 2) {
      for (int j = 0; j < thread_size; j += 2) {
        std::vector<double> C_temp(no_of_elements * 4, 0);
        const int rank = (thread_size)*i + j;

        MPI_Status stat;
        MPI_Recv(&C_temp.at(0), no_of_elements * 4, MPI_DOUBLE, rank, 1000,
                 MPI_COMM_WORLD, &stat);

        for (int k = 0; k < rows * 2; k++) {
          for (int l = 0; l < rows * 2; l++) {
            C[i * thread_size * size + j * thread_size + size * k + l] =
                C_temp.at(k * rows * 2 + l);
          }
        }
      }
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
}
