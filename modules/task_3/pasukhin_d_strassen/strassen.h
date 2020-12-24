// Copyright 2020 Pasukhin Dmitry

#ifndef MODULES_TASK_3_PASUKHIN_D_STRASSEN_STRASSEN_H_
#define MODULES_TASK_3_PASUKHIN_D_STRASSEN_STRASSEN_H_

void GenerateMatrix(double* matrix, const int size, const int seed);
void Strassen_Multiplication(const double* A, const double* B, double* C, const int size);
void Serial_Multiply(const double* A, const double* B, double* C, const int size);
void Serial_Strassen(const double* A, const double* B, double* C, const int size);

#endif  // MODULES_TASK_3_PASUKHIN_D_STRASSEN_STRASSEN_H_
