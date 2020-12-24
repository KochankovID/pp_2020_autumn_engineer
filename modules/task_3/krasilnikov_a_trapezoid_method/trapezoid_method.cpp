// Copyright 2020 Krasilnikov Alexey

#include <mpi.h>
#include <iostream>
#include <ctime>
#include <cmath>
#include <random>
#include <algorithm>
#include <vector>
#include <utility>
#include "../../../modules/task_3/krasilnikov_a_trapezoid_method/trapezoid_method.h"

double trapezoidMethodSequential(double (*f)(double, double), double a_x, double b_x, double a_y, double b_y) {
  const int n = 10000;
  double ans = 0;
  double dx = (b_x - a_x) / n;
  double dy = (b_y - a_y) / n;

  for (double x = a_x + dx; std::abs(x - b_x) > 0.00000001; x += dx) {
    double cur_ans = (f(x, a_y) + f(x, b_y)) / 2;
    for (double y = a_y + dy; std::abs(y - b_y) > 0.00000001; y += dy) {
      cur_ans += f(x, y);
    }
    ans += cur_ans;
  }

  ans += (f(a_x, a_y) + f(a_x, b_y) + f(b_x, a_y) + f(b_x, b_y)) / 4;
  for (double y = a_y + dy; std::abs(y - b_y) > 0.00000001; y += dy) {
    ans += f(a_x, y) / 2;
    ans += f(b_x, y) / 2;
  }

  return ans * dx * dy;
}

double trapezoidMethodParallel(double (*f)(double, double), double a_x, double b_x, double a_y, double b_y) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const int n = 10000;
  double ans;
  double local_ans = 0;
  double dx = (b_x - a_x) / n;
  double dy = (b_y - a_y) / n;

  double x;
  if (rank == 0) {
    x = a_x + dx * size;
  } else {
    x = a_x + dx * rank;
  }
  for (; b_x - x > 0.00000001; x += dx * size) {
    double cur_ans = (f(x, a_y) + f(x, b_y)) / 2;
    for (double y = a_y + dy; std::abs(y - b_y) > 0.00000001; y += dy) {
      cur_ans += f(x, y);
    }
    local_ans += cur_ans;
  }
  MPI_Reduce(&local_ans, &ans, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    ans += (f(a_x, a_y) + f(a_x, b_y) + f(b_x, a_y) + f(b_x, b_y)) / 4;
    for (double y = a_y + dy; std::abs(y - b_y) > 0.00000001; y += dy) {
      ans += f(a_x, y) / 2;
      ans += f(b_x, y) / 2;
    }
  }
  return ans * dx * dy;
}
