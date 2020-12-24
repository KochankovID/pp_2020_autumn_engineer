// Copyright 2020 Sozinov Alex
#ifndef MODULES_TASK_3_SOZINOV_A_SPARSE_MATRIX_COL_SPARSE_MATRIX_COL_H_
#define MODULES_TASK_3_SOZINOV_A_SPARSE_MATRIX_COL_SPARSE_MATRIX_COL_H_

#include <vector>
#include <iostream>

struct ComprColStorage {
  int countRow;
  int countCol;
  std::vector<double> value;
  std::vector<int> startPos;  // starting vector position (vector value) for i column
  std::vector<int> numberRow;  // number Row for element std::vector value

  bool operator==(const ComprColStorage& sec) const;
  bool operator!=(const ComprColStorage& sec) const;
};

ComprColStorage GenerateMatrix(int countRow, int countCol, double proc);
ComprColStorage GetSequentialSolution(ComprColStorage *first, ComprColStorage *second);
ComprColStorage GetParalSolution(ComprColStorage *first, ComprColStorage *second);
double Sum(const ComprColStorage &first, const ComprColStorage &second, int curRow, int curCol, int size);
void PrintMatr(const ComprColStorage &matr);
#endif  // MODULES_TASK_3_SOZINOV_A_SPARSE_MATRIX_COL_SPARSE_MATRIX_COL_H_
