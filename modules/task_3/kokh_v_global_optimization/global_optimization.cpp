// Copyright 2020 Kokh Vladislav
#include <mpi.h>
#include <stdexcept>
#include <set>
#include <random>
#include "../../../modules/task_3/kokh_v_global_optimization/global_optimization.h"

double func1(double x, double y) {
    return std::pow(y - 1, 2) + std::pow(x, 2);
}

double func2(double x, double y) {
    return std::pow(std::pow(std::pow(x, 2) + std::pow(y, 2), 2), 1.0 / 3) + 4;
}

double func3(double x, double y) {
    return std::pow(x, 3) + 8*std::pow(y, 3) - 6*x*y + 5;
}
double func4(double x, double y) {
    return y*sqrt(x) - 2*std::pow(y, 2) - x + 14*y;
}
double func5(double x, double y) {
    return x + 4*y - 6 - 2*log(x*y) - 3*log(y);
}


xyzStruct twoPar(const double& a_first, const double& b_first, const double& a_second,
const double& b_second, double(*func)(double x, double y), const double& eps_in_func,
const int& n_max_value, const double& eps_one_value,
const int& n_max_value_1, const double& r_param) {
    if (a_first > b_first)
        throw "a_first > b_first";

    if (a_second > b_second)
        throw "a_second > b_second";

    int rank, size;
    MPI_Status status;
    xyzStruct result;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    xyzStruct last_result = { 0, 0, 0 };

    std::set<setR> rEl;
    double dop_param, current_dop_param, dop_param_2, current_r, new_x;
    bool term_value = false;

    if (size < 2)
        return twoParSequential(a_first, b_first, a_second, b_second, func);
    if (rank == 0) {
        std::set<setTwo> set;
        int k;

        if (b_first - a_first <= 0.0001 || b_second - a_second <= 0.0001) {
            k = size + 1;
        } else {
            k = size;
        }
        double len_of_part = (b_first - a_first) / (k - 1);

        for (int i = 0; i < size - 1; ++i) {
            double x_val_send = a_first + i * len_of_part;
            MPI_Send(&x_val_send, 1, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
        }

        result = forOne(a_second, b_second, b_first, func, eps_one_value);
        set.insert(setTwo(result.x, result.y, result.z));
        last_result = result;

        if (k != size) {
            result = forOne(a_second, b_second, a_first + len_of_part * size, func, eps_one_value);
            set.insert(setTwo(result.x, result.y, result.z));
            if (result.z < last_result.z) { last_result = result;}
        }

        for (int j = 0; j < size - 1; ++j) {
            MPI_Recv(&result, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            if (result.z < last_result.z) { last_result = result; }
            set.insert(setTwo(result.x, result.y, result.z));
        }

        while (!term_value && k < n_max_value) {
            rEl.clear();
            dop_param = -1;
            auto i_value = set.begin();
            i_value++;
            auto i_previous_value = set.begin();

            while (i_value != set.end()) {
                current_dop_param = std::abs(static_cast<double>((i_value->z - i_previous_value->z) /
                (i_value->x - i_previous_value->x)));
                if (current_dop_param > dop_param) { dop_param = current_dop_param; }
                i_value++; i_previous_value++;
            }
            if (dop_param > 0) {
                dop_param_2 = r_param * dop_param;
            } else { dop_param_2 = 1; }

            i_value = set.begin(); i_value++;
            i_previous_value = set.begin();

            while (i_value != set.end()) {
                double ff = dop_param_2 * (i_value->x - i_previous_value->x);
                double sspow = std::pow((i_value->z - i_previous_value->z), 2) /
                (dop_param_2 * (i_value->x - i_previous_value->x));
                double value_minus = 2 * (i_value->z - i_previous_value->z);
                current_r =  ff + sspow - value_minus;

                rEl.insert(setR(current_r, i_value->x, i_value->z, i_previous_value->x,
                i_previous_value->z));
                i_value++; i_previous_value++;
            }

            auto itR = rEl.begin();

            for (int i = 0; i < size - 1; ++i) {
                k++;
                new_x = 0.5 * (itR->x + itR->xPrev) - ((itR->z - itR->zPrev) / (2 * dop_param_2));
                MPI_Send(&new_x, 1, MPI_DOUBLE, i+1, 1, MPI_COMM_WORLD);
                if (itR->x - itR->xPrev <= eps_in_func) { term_value = true;}
                itR++;
            }
            for (int i = 0; i < size - 1; ++i) {
                MPI_Recv(&result, 3, MPI_DOUBLE, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
                if (result.z < last_result.z) { last_result = result;}
                set.insert(setTwo(result.x, result.y, result.z));
            }
        }


        for (int i = 0; i < size - 1; ++i) {
            double term_value = eps_in_func  * 0.001;
            MPI_Send(&term_value, 1, MPI_DOUBLE, i+1, 1, MPI_COMM_WORLD);
        }
    } else {
        bool term_value = false;
        while (!term_value) {
            double mes;
            MPI_Recv(&mes, 1, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &status);
            if (mes == eps_in_func * 0.001) {
               term_value = true;
            } else {
                result = forOne(a_second, b_second, mes, func, eps_one_value, n_max_value_1);
                MPI_Send(&result, 3, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
            }
        }
    }
    return last_result;
}

xyzStruct twoParSequential(const double& a_first, const double& b_first,
const double& a_second, const double& b_second,
double(*func)(double x, double y), const double& eps_in_func, const int& n_max_value, const double& eps_one_value,
const int& n_max_value_1, const double& r_param) {
    if (a_first > b_first)
        throw "a_first > b_first";

    if (a_second > b_second)
        throw "a_second > b_second";

    xyzStruct result;
    xyzStruct last_result = {0, 0, 0};
    std::set<setTwo> set;

    result = forOne(a_second, b_second, a_first, func, eps_one_value, n_max_value_1);
    set.insert(setTwo(result.x, result.y, result.z));
    last_result = result;

    result = forOne(a_second, b_second, b_first, func, eps_one_value, n_max_value_1);
    set.insert(setTwo(result.x, result.y, result.z));

    if (result.z < last_result.z) { last_result = result;}
    double dop_param, current_dop_param, dop_param_2, current_r, new_x;
    int k = 2;
    std::set<setR> rEl;
    bool term_value = false;

    while (!term_value && k < n_max_value) {
        rEl.clear();
        dop_param = -1;
        auto i_value = set.begin();
        i_value++;
        auto i_previous_value = set.begin();

        while (i_value != set.end()) {
            current_dop_param = std::abs(static_cast<double>((i_value->z - i_previous_value->z) /
            (i_value->x - i_previous_value->x)));
            if (current_dop_param > dop_param) { dop_param = current_dop_param; }
            i_value++; i_previous_value++;
        }
        if (dop_param > 0) {
            dop_param_2 = r_param * dop_param;
        } else {dop_param_2 = 1;}

        i_value = set.begin();
        i_value++;
        i_previous_value = set.begin();

        while (i_value != set.end()) {
            double ff = dop_param_2 * (i_value->x - i_previous_value->x);
            double sspow = std::pow((i_value->z - i_previous_value->z), 2) /
            (dop_param_2 * (i_value->x - i_previous_value->x));
            double value_minus = 2 * (i_value->z - i_previous_value->z);
            current_r = ff + sspow - value_minus;
            rEl.insert(setR(current_r, i_value->x, i_value->z, i_previous_value->x, i_previous_value->z));
            i_value++; i_previous_value++;
        }

        k++;
        auto Riter = rEl.begin();
        new_x = 0.5 * (Riter->x + Riter->xPrev) - ((Riter->z - Riter->zPrev) / (2 * dop_param_2));
        result = forOne(a_second, b_second, new_x, func, eps_one_value, n_max_value_1);
        set.insert(setTwo(result.x, result.y, result.z));

        if (result.z < last_result.z) {last_result = result;}
        if (Riter->x - Riter->xPrev <= eps_in_func) {term_value = true;}
    }
    return last_result;
}

xyzStruct forOne(const double& a_value, const double& b_value,
const double& x_val_send, double(*func)(double x, double y),
const double& eps_in_func, const int& n_max_value, const double& r_param) {
    if (a_value > b_value)
        throw "a_value > b_value";

    xyzStruct result;
    bool st_flag = false;
    std::set<setOne> set;

    double dop_param, current_dop_param, dop_param_2, R, current_r, new_x;
    double dop_valuel_forOne = func(x_val_send, a_value);

    result.x = x_val_send;

    set.insert(setOne(a_value, dop_valuel_forOne));
    result.y = a_value;
    result.z = dop_valuel_forOne;
    dop_valuel_forOne = func(x_val_send, b_value);
    set.insert(setOne(b_value, dop_valuel_forOne));

    if (result.z > dop_valuel_forOne) {
        result.y = b_value;
        result.z = dop_valuel_forOne;
    }

    int k = 2;
    auto maxRiter = set.begin();
    auto r_i_value_previous_max = set.begin();

    while (!st_flag && k < n_max_value) {
        dop_param = -1;
        auto i_value = set.begin();
        i_value++;
        auto i_previous_value = set.begin();

        while (i_value != set.end()) {
            current_dop_param = std::abs(static_cast<double>((i_value->y - i_previous_value->y) /
            (i_value->x - i_previous_value->x)));
            if (current_dop_param > dop_param) {dop_param = current_dop_param;}
            i_value++; i_previous_value++;
        }
        if (dop_param > 0) {
            dop_param_2 = r_param * dop_param;
        } else {dop_param_2 = 1;}

        i_value = set.begin();
        i_value++;
        i_previous_value = set.begin();
        R = -200000000;

        while (i_value != set.end()) {
            double ff = dop_param_2 * (i_value->x - i_previous_value->x);
            double sspow = std::pow((i_value->y - i_previous_value->y), 2) /
            (dop_param_2 * (i_value->x - i_previous_value->x));
            double value_minus = 2 * (i_value->y - i_previous_value->y);
            current_r = ff + sspow - value_minus;

            if (current_r > R) {
                R = current_r;
                maxRiter = i_value;
                r_i_value_previous_max = i_previous_value;
            }

            i_value++; i_previous_value++;
        }

        k++;
        new_x = 0.5 * (maxRiter->x + r_i_value_previous_max->x) -
        ((maxRiter->y - r_i_value_previous_max->y) / (2 * dop_param_2));
        dop_valuel_forOne = func(x_val_send, new_x);
        set.insert(setOne(new_x, dop_valuel_forOne));

        if (result.z > dop_valuel_forOne) {
            result.y = new_x;
            result.z = dop_valuel_forOne;
        }

        if (maxRiter->x - r_i_value_previous_max->x <= eps_in_func) {st_flag = true;}
    }
    return result;
}

bool comparingResults(const xyzStruct& a, const xyzStruct& b, const double& eps_value) {
    bool eq = false;
    if (std::abs(static_cast<double>(a.x - b.x)) <= eps_value) {
        if (std::abs(static_cast<double>(a.y - b.y)) <= eps_value) {
            if (std::abs(static_cast<double>(a.z - b.z)) <= eps_value)
                eq = true;
        }
    }
    return eq;
}
