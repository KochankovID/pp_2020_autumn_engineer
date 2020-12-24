// Copyright 2020 Igor Rukhovich
#include "../../modules/task_1/rukhovich_i_integration_trapezoidal_method/integration_trapezoidal_method.h"
#include <mpi.h>
#include <cmath>
#include <memory>
#include <stdexcept>


QuadraticPolynomial::QuadraticPolynomial(double a, double b, double c) : a(a), b(b), c(c) {}

double QuadraticPolynomial::operator()(double point) {
    return a*point*point + b*point + c;
}

double getIntegralSequentional(std::shared_ptr<Function> func, double from, double to,
                               double precision) {
    if (to < from) {
        throw std::runtime_error("'from' must not be greater than 'to'");
    }
    if (precision <= 0.) {
        throw std::runtime_error("'precision' must be greater than 0");
    }

    double cur_val = (*func)(from), cur_point = from, ans = 0.;
    if (cur_val < 0.) {
        throw std::runtime_error("Function value must not be less than 0");
    }
    while (cur_point + precision < to) {
        cur_point += precision;

        double next_val = (*func)(cur_point);
        if (next_val < 0.) {
            throw std::runtime_error("Function value must not be less than 0");
        }
        ans += (cur_val + next_val) * precision / 2.;

        cur_val = next_val;
    }

    double last_val = (*func)(to);
    if (last_val < 0.) {
        throw std::runtime_error("Function value must not be less than 0");
    }
    ans += (cur_val + last_val) * (to - cur_point) / 2.;
    return ans;
}

double getIntegralParallel(std::shared_ptr<Function> func, double from, double to,
                           double precision) {
    int rank, num_proc;
    MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        if (to < from) {
            throw std::runtime_error("'from' must not be greater than 'to'");
        }
        if (precision <= 0.) {
            throw std::runtime_error("'precision' must be greater than 0");
        }

        uint64_t num_iters_total = std::ceil((to - from) / precision);
        uint64_t num_iters_each = num_iters_total / static_cast<uint64_t>(num_proc);
        to -= num_iters_each * precision * static_cast<uint64_t>(num_proc - 1);

        for (size_t i = 1; i < static_cast<uint64_t>(num_proc); ++i) {
            double local_from = to + num_iters_each * precision * (i - 1);
            double local_to = local_from + num_iters_each * precision;
            MPI_Send(&local_from, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            MPI_Send(&local_to, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Status status;
        MPI_Recv(&from, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        MPI_Recv(&to, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
    }
    MPI_Bcast(&precision, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_result = getIntegralSequentional(func, from, to, precision);
    double ans = 0;
    MPI_Reduce(&local_result, &ans, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return ans;
}
