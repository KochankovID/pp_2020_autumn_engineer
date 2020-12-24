// Copyright 2020 Evseev Alexander
#include <mpi.h>
#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <vector>

#include "../../../modules/task_3/evseev_a_intergral_by_mont_carl/integral_by_mont_carl.h"

double MonteCarloSequential(
    std::vector<double> a, double sd,
    double (*pfunction)(std::vector<double>), bool (*parea)(std::vector<double>),
    unsigned int dim, int p_count) {
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  std::uniform_real_distribution<> urd(0, 1);

  std::vector<double> p(dim * p_count);
  for (unsigned int i = 0; i < p.size(); i++) {
    p[i] = a[i % dim] + urd(gen) * sd;
  }

  int n = 0;
  double val = 0;

  for (int i = 0; i < p_count; ++i) {
    std::vector<double> cpoint;
    for (unsigned int j = 0; j < dim; ++j) {
      cpoint.push_back(p[dim * i + j]);
    }
    if (parea(cpoint)) {
      n++;
      val += pfunction(cpoint);
    }
  }

  val = val / n;
  double area = std::pow(sd, dim) * n / p_count;

  return area * val;
}

double MonteCarloParallel(std::vector<double> a, double sd,
                                   double (*pfunction)(std::vector<double>),
                                   bool (*parea)(std::vector<double>),
                                   unsigned int dim, int p_count) {
  if (a.size() != dim) throw(1);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int delta = p_count / size;
  std::vector<double> p(dim * p_count);

  if (rank == 0) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(NULL)));
    std::uniform_real_distribution<> urd(0, 1);
    for (unsigned int i = 0; i < p.size(); i++) {
      p[i] = a[i % dim] + urd(gen) * sd;
    }
  }

  std::vector<double> loc_p(delta * dim);

  MPI_Scatter(&p[0], delta * dim, MPI_DOUBLE, &loc_p[0],
              delta * dim, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  int loc_n = 0;
  int n = 0;
  double loc_val = 0;
  double val = 0;

  for (int i = 0; i < delta; ++i) {
    std::vector<double> cpoint;
    for (unsigned int j = 0; j < dim; ++j) {
      cpoint.push_back(loc_p[dim * i + j]);
    }
    if (parea(cpoint)) {
      loc_n++;
      loc_val += pfunction(cpoint);
    }
  }
  MPI_Reduce(&loc_n, &n, 1, MPI_INT, MPI_SUM, 0,
             MPI_COMM_WORLD);
  MPI_Reduce(&loc_val, &val, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  val = val / n;
  double area = std::pow(sd, dim) * n / p_count;

  return area * val;
}
// file deepcode ignore DivisionByZero: <comment the reason here>
