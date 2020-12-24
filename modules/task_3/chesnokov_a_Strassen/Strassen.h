// Copyright 2020 Chesnokov
#ifndef MODULES_TASK_3_CHESNOKOV_A_STRASSEN_STRASSEN_H_
#define MODULES_TASK_3_CHESNOKOV_A_STRASSEN_STRASSEN_H_

class Matrix {
 public:
    double* buf;
    int columns, rows;

    Matrix() : buf(nullptr), columns(0), rows(0) {}
    Matrix(int col, int r);
    Matrix(int col, int r, double* buf);
    Matrix(const Matrix& m);
    ~Matrix() {
        delete[] buf;
    }
    Matrix operator+(const Matrix & b);
    Matrix operator-(const Matrix & b);
    Matrix& operator=(const Matrix & m);
};

Matrix getRandomMatrix(int columns, int rows);
Matrix getMatrixMul(const Matrix& a, const Matrix& b);
Matrix getMatrixMulStrassen(const Matrix& a, const Matrix& b);
Matrix getParallelMatrixMul(const Matrix& a, const Matrix& b);

// Helper functions------------------------------------------

// returns quarter sub matrix with coords (x,y) = {1,2}^2
void GetQuarterOf(Matrix * recv, const Matrix & src, int y, int x);
// sets quarter in matrix to coords(x, y) = { 1,2 } ^ 2
void SetQuarterTo(Matrix * recv, const Matrix & src, int y, int x);

// makes matrix from quarters
Matrix assembleMatrix(const Matrix & c11, const Matrix & c12, const Matrix & c21, const Matrix & c22);

// rounds matrix dimensions to be equal and multiple of m with extra 0-filled rows and columns
Matrix RoundToSquareMatrix(const Matrix & mat, int m);
Matrix RoundToSpecificSquareMatrix(const Matrix & mat, int size);
// shrinks matrix to upper left c-columns and r-rows
Matrix ShrinkMatrix(const Matrix & mat, int c, int r);

#endif  // MODULES_TASK_3_CHESNOKOV_A_STRASSEN_STRASSEN_H_
