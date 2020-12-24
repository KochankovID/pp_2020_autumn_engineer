// Copyright 2020 Chesnokov
#include "../../../modules/task_3/chesnokov_a_Strassen/Strassen.h"
#include <mpi.h>
#include <random>
#include <ctime>
#include <iostream>

using std::cout;
using std::endl;

Matrix::Matrix(int col, int r) : columns(col), rows(r) {
    buf = new double[col * r];
    for (int i = 0; i < col * r; i++) {
        buf[i] = 0.0;
    }
}

Matrix::Matrix(int col, int r, double * buf) : columns(col), rows(r) {
    this->buf = new double[col * r];
    for (int i = 0; i < col * r; i++) {
        this->buf[i] = buf[i];
    }
}

Matrix::Matrix(const Matrix & m) : columns(m.columns), rows(m.rows) {
    this->buf = new double[columns * rows];
    for (int i = 0; i < columns * rows; i++) {
        this->buf[i] = m.buf[i];
    }
}

Matrix Matrix::operator+(const Matrix & b) {
    if (columns != b.columns || rows != b.rows) {
        cout << "[ERROR] invalid dimensions" << endl;
        cout << "columns = " << columns << " b.columns = " << b.columns << endl;
        cout << "rows = " << rows << " b.rows = " << b.rows << endl;
        throw "Invalid dimension";
    }
    Matrix res(*this);
    for (int i = 0; i < res.columns * res.rows; i++) {
        res.buf[i] += b.buf[i];
    }
    return res;
}

Matrix Matrix::operator-(const Matrix & b) {
    if (columns != b.columns || rows != b.rows) {
        cout << "[ERROR] invalid dimensions" << endl;
        cout << "columns = " << columns << " b.columns = " << b.columns << endl;
        cout << "rows = " << rows << " b.rows = " << b.rows << endl;
        throw "Invalid dimension";
    }
    Matrix res(*this);
    for (int i = 0; i < res.columns * res.rows; i++) {
        res.buf[i] -= b.buf[i];
    }
    return res;
}

Matrix & Matrix::operator=(const Matrix & m) {
    if (this == &m) return *this;
    if (buf) delete[] buf;
    columns = m.columns;
    rows = m.rows;
    buf = new double[columns * rows];
    for (int i = 0; i < columns * rows; i++) {
        buf[i] = m.buf[i];
    }
    return *this;
}


Matrix getRandomMatrix(int columns, int rows) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));
    Matrix matrix(columns, rows);
    for (int i = 0; i < columns; i++) {
        for (int j = 0; j < rows; j++) {
            matrix.buf[i * rows + j] = gen() % 10;
        }
    }
    return matrix;
}

Matrix getMatrixMul(const Matrix & a, const Matrix & b) {
    if (a.columns != b.rows) {
        cout << "[ERROR] invalid dimensions" << endl;
        cout << "a.columns = " << a.columns << " b.rows = " << b.rows << endl;
        throw "Invalid dimensions";
    }
    Matrix res(b.columns, a.rows);
    for (int i = 0; i < res.rows; i++)
        for (int j = 0; j < res.columns; j++)
            for (int r = 0; r < a.columns; r++) {
                res.buf[i * res.columns + j] +=
                    a.buf[i * a.columns + r] * b.buf[r * b.columns + j];
            }
    return res;
}

Matrix getMatrixMulStrassen(const Matrix & a, const Matrix & b) {
    if (a.columns <= 64 || a.rows <= 64) return getMatrixMul(a, b);

    // resizing
    Matrix _a = RoundToSquareMatrix(a, 64);
    Matrix _b = RoundToSquareMatrix(b, 64);
    if (_a.columns > _b.columns) {
        _b = RoundToSpecificSquareMatrix(_b, _a.columns);
    } else if (_a.columns < _b.columns) {
        _a = RoundToSpecificSquareMatrix(_a, _b.columns);
    }

    Matrix A11(_a.columns / 2, _a.rows / 2);
    Matrix A12(_a.columns / 2, _a.rows / 2);
    Matrix A21(_a.columns / 2, _a.rows / 2);
    Matrix A22(_a.columns / 2, _a.rows / 2);

    Matrix B11(_b.columns / 2, _b.rows / 2);
    Matrix B12(_b.columns / 2, _b.rows / 2);
    Matrix B21(_b.columns / 2, _b.rows / 2);
    Matrix B22(_b.columns / 2, _b.rows / 2);

    GetQuarterOf(&A11, _a, 1, 1);
    GetQuarterOf(&A12, _a, 1, 2);
    GetQuarterOf(&A21, _a, 2, 1);
    GetQuarterOf(&A22, _a, 2, 2);

    GetQuarterOf(&B11, _b, 1, 1);
    GetQuarterOf(&B12, _b, 1, 2);
    GetQuarterOf(&B21, _b, 2, 1);
    GetQuarterOf(&B22, _b, 2, 2);

    Matrix P1 = getMatrixMulStrassen(A11 + A22, B11 + B22);
    Matrix P2 = getMatrixMulStrassen(A21 + A22, B11);
    Matrix P3 = getMatrixMulStrassen(A11, B12 - B22);
    Matrix P4 = getMatrixMulStrassen(A22, B21 - B11);
    Matrix P5 = getMatrixMulStrassen(A11 + A12, B22);
    Matrix P6 = getMatrixMulStrassen(A21 - A11, B11 + B12);
    Matrix P7 = getMatrixMulStrassen(A12 - A22, B21 + B22);

    Matrix C11 = P1 + P4 - P5 + P7;
    Matrix C12 = P3 + P5;
    Matrix C21 = P2 + P4;
    Matrix C22 = P1 - P2 + P3 + P6;

    return ShrinkMatrix(assembleMatrix(C11, C12, C21, C22), a.rows, b.columns);
}

Matrix getParallelMatrixMul(const Matrix & a, const Matrix & b) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Resizing
    Matrix _a = RoundToSquareMatrix(a, 64);
    Matrix _b = RoundToSquareMatrix(b, 64);
    if (_a.columns > _b.columns) {
        _b = RoundToSpecificSquareMatrix(_b, _a.columns);
    } else if (_a.columns < _b.columns) {
        _a = RoundToSpecificSquareMatrix(_a, _b.columns);
    }

    int lvl;
    int mult_count;
    for (mult_count = 7, lvl = 1; mult_count < size; mult_count *= 7, lvl++) {}


    // calculating delta
    int delta = mult_count / size;
    int tail = mult_count % size;
    // num of mults
    int count;
    if (rank < tail) {
        count = delta + 1;
    } else {
        count = delta;
    }

    // Work division
    if (rank == 0) {
        Matrix ** matrices = new Matrix*[mult_count];
        for (int i = 0; i < mult_count; i++) {
            matrices[i] = new Matrix[2];
        }

        matrices[0][0] = _a; matrices[0][1] = _b;

        int mults = 1;
        int stride = mult_count;
        for (int i = 0; i < lvl; i++) {
            for (int j = 0; j < mults; j++) {
                int c = matrices[j * stride][0].columns / 2;
                int r = matrices[j * stride][0].rows / 2;
                Matrix A11(c, r);
                Matrix A12(c, r);
                Matrix A21(c, r);
                Matrix A22(c, r);

                c = matrices[j * stride][1].columns / 2;
                r = matrices[j * stride][1].rows / 2;
                Matrix B11(c, r);
                Matrix B12(c, r);
                Matrix B21(c, r);
                Matrix B22(c, r);

                GetQuarterOf(&A11, matrices[j*stride][0], 1, 1);
                GetQuarterOf(&A12, matrices[j*stride][0], 1, 2);
                GetQuarterOf(&A21, matrices[j*stride][0], 2, 1);
                GetQuarterOf(&A22, matrices[j*stride][0], 2, 2);

                GetQuarterOf(&B11, matrices[j*stride][1], 1, 1);
                GetQuarterOf(&B12, matrices[j*stride][1], 1, 2);
                GetQuarterOf(&B21, matrices[j*stride][1], 2, 1);
                GetQuarterOf(&B22, matrices[j*stride][1], 2, 2);

                matrices[j*stride + 0 * (stride / 7)][0] = A11 + A22;
                matrices[j*stride + 1 * (stride / 7)][0] = A21 + A22;
                matrices[j*stride + 2 * (stride / 7)][0] = A11;
                matrices[j*stride + 3 * (stride / 7)][0] = A22;
                matrices[j*stride + 4 * (stride / 7)][0] = A11 + A12;
                matrices[j*stride + 5 * (stride / 7)][0] = A21 - A11;
                matrices[j*stride + 6 * (stride / 7)][0] = A12 - A22;

                matrices[j*stride + 0 * (stride / 7)][1] = B11 + B22;
                matrices[j*stride + 1 * (stride / 7)][1] = B11;
                matrices[j*stride + 2 * (stride / 7)][1] = B12 - B22;
                matrices[j*stride + 3 * (stride / 7)][1] = B21 - B11;
                matrices[j*stride + 4 * (stride / 7)][1] = B22;
                matrices[j*stride + 5 * (stride / 7)][1] = B11 + B12;
                matrices[j*stride + 6 * (stride / 7)][1] = B21 + B22;
            }
            mults *= 7; stride /= 7;
        }

        int m_size = matrices[0][0].columns;
        MPI_Bcast(&m_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        int proc;
        for (proc = 1; proc < tail; proc++) {
            for (int i = 0; i < delta + 1; i++) {
                MPI_Send(matrices[proc*(delta + 1) + i][0].buf, m_size * m_size, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
                MPI_Send(matrices[proc*(delta + 1) + i][1].buf, m_size * m_size, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD);
            }
        }
        for (; proc < size; proc++) {
            for (int i = 0; i < delta; i++) {
                MPI_Send(matrices[tail*(delta + 1) + (proc - tail)*delta + i][0].buf,
                    m_size * m_size, MPI_DOUBLE, proc, 1, MPI_COMM_WORLD);
                MPI_Send(matrices[tail*(delta + 1) + (proc - tail)*delta + i][1].buf,
                    m_size * m_size, MPI_DOUBLE, proc, 2, MPI_COMM_WORLD);
            }
        }

        // gathering results mults
        Matrix * P = new Matrix[mult_count];
        // local 0th rank computations
        for (int i = 0; i < count; i++) {
            P[i] = getMatrixMulStrassen(matrices[i][0], matrices[i][1]);
        }
        MPI_Status status;
        for (proc = 1; proc < tail; proc++) {
            for (int i = 0; i < delta + 1; i++) {
                P[proc*(delta + 1) + i] = Matrix(m_size, m_size);
                MPI_Recv(P[proc*(delta + 1) + i].buf, m_size * m_size, MPI_DOUBLE, proc, 3, MPI_COMM_WORLD, &status);
            }
        }
        for (; proc < size; proc++) {
            for (int i = 0; i < delta; i++) {
                P[tail*(delta + 1) + (proc - tail)*delta + i] = Matrix(m_size, m_size);
                MPI_Recv(P[tail*(delta + 1) + (proc - tail)*delta + i].buf, m_size * m_size,
                    MPI_DOUBLE, proc, 3, MPI_COMM_WORLD, &status);
            }
        }

        // calculating final result
        mults = mult_count / 7;  stride = 1;
        for (int i = 0; i < lvl; i++) {
            for (int j = 0; j < mults; j++) {
                Matrix C11 = P[j*stride * 7 + 0 * stride] + P[j*stride * 7 + 3 * stride]
                    - P[j*stride * 7 + 4 * stride] + P[j*stride * 7 + 6 * stride];
                Matrix C12 = P[j*stride * 7 + 2 * stride] + P[j*stride * 7 + 4 * stride];
                Matrix C21 = P[j*stride * 7 + 1 * stride] + P[j*stride * 7 + 3 * stride];
                Matrix C22 = P[j*stride * 7 + 0 * stride] - P[j*stride * 7 + 1 * stride]
                    + P[j*stride * 7 + 2 * stride] + P[j*stride * 7 + 5 * stride];
                P[j * stride * 7] = assembleMatrix(C11, C12, C21, C22);
            }
            mults /= 7; stride *= 7;
        }

        return ShrinkMatrix(P[0], a.rows, b.columns);

    } else {  // All process with rank != 0 just count received mults
        int m_size;
        MPI_Bcast(&m_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // local sub matrices
        Matrix * l_a = new Matrix[count];
        Matrix * l_b = new Matrix[count];
        MPI_Status status;
        for (int i = 0; i < count; i++) {
            l_a[i] = Matrix(m_size, m_size);
            MPI_Recv(l_a[i].buf, m_size * m_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            l_b[i] = Matrix(m_size, m_size);
            MPI_Recv(l_b[i].buf, m_size * m_size, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD, &status);
        }
        Matrix * mults = new Matrix[count];
        for (int i = 0; i < count; i++) {
            mults[i] = getMatrixMulStrassen(l_a[i], l_b[i]);
            MPI_Send(mults[i].buf, m_size * m_size, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD);
        }

        return Matrix();
    }
}

void GetQuarterOf(Matrix * recv, const Matrix & src, int y, int x) {
    x--; y--;
    int c2 = src.columns / 2;
    int r2 = src.rows / 2;
    for (int i = 0; i < r2; i++)
        for (int j = 0; j < c2; j++) {
            recv->buf[i * c2 + j] = src.buf[src.columns*r2*y + x*c2 + i * c2 * 2 + j];
        }
}

void SetQuarterTo(Matrix * recv, const Matrix & src, int y, int x) {
    x--; y--;
    int c = src.columns;
    int r = src.rows;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            recv->buf[r*c*2*y + x*c + i*c*2 + j] = src.buf[i*c + j];
        }
}

Matrix assembleMatrix(const Matrix & c11, const Matrix & c12, const Matrix & c21, const Matrix & c22) {
    Matrix res(c11.rows + c21.rows, c11.columns + c12.columns);
    SetQuarterTo(&res, c11, 1, 1);
    SetQuarterTo(&res, c12, 1, 2);
    SetQuarterTo(&res, c21, 2, 1);
    SetQuarterTo(&res, c22, 2, 2);
    return res;
}

Matrix RoundToSquareMatrix(const Matrix & mat, int m) {
    if (mat.columns % m == 0 && mat.rows % m == 0 && mat.rows == mat.columns) return mat;
    int max;
    if (mat.columns % m == 0 && mat.rows % m == 0) {
        max = mat.columns > mat.rows ? mat.columns : mat.rows;
    } else {
        int new_c, new_r;
        new_c = (mat.columns / m + 1) * m;
        new_r = (mat.rows / m + 1) * m;
        if (new_c > new_r) {
            max = new_c;
        } else {
            max = new_r;
        }
    }

    Matrix result(max, max);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.columns; j++) {
            result.buf[i*max + j] = mat.buf[i*mat.columns + j];
        }
    }
    return result;
}

Matrix RoundToSpecificSquareMatrix(const Matrix & mat, int size) {
    Matrix result(size, size);
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.columns; j++) {
            result.buf[i*size + j] = mat.buf[i*mat.columns + j];
        }
    }
    return result;
}

Matrix ShrinkMatrix(const Matrix & mat, int c, int r) {
    if (mat.columns == c && mat.rows == r) {
        return mat;
    }
    Matrix result(c, r);
    for (int i = 0; i < c; i++) {
        for (int j = 0; j < r; j++) {
            result.buf[i*r + j] = mat.buf[i*mat.rows + j];
        }
    }
    return result;
}
