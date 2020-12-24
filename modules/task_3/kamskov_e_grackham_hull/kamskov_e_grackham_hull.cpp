// Copyright 2020 Kamskov Evgeny
#include <mpi.h>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <cstring>
#include "../../../modules/task_3/kamskov_e_grackham_hull/kamskov_e_grackham_hull.h"

Point first;

struct Comparator {
    bool operator () (Point a, Point b) {
        if (rotation(first, a, b) > 0)
            return true;
        else if (rotation(first, a, b) < 0)
            return false;
        else
            return distance(first, b, a);
    }
};

std::vector<Point> greckham_seq(std::vector < Point > p) {
    std::vector <int> positions;
    int n = p.size();
    int i, k = 0;
    first = p[0];
    for (i = 1; i < n; ++i)
        if (first.x > p[i].x || (first.x == p[i].x && first.y > p[i].y)) {
            first = p[i];
            k = i;
        }
    Point end = first;
    for (i = 1; i < n; ++i)
        if (end.x < p[i].x || (end.x == p[i].x && end.y < p[i].y)) {
            end = p[i];
        }

    std::swap(p[0], p[k]);
    sort(p.begin() + 1, p.end(), Comparator());

    positions.push_back(0);
    positions.push_back(1);
    i = 2;
    while (i < n) {
        if (positions.size() < 2) {
            positions.push_back(i);
            ++i;
        }
        if (rotation(p[positions[positions.size() - 2]], p[positions[positions.size() - 1]], p[i]) <= 0) {
            positions.pop_back();
        } else {
            positions.push_back(i);
            ++i;
        }
    }

    std::vector<Point> vec(positions.size());
    for (size_t i = 0; i < positions.size(); i++) {
        vec[i] = p[positions[i]];
    }
    end = first;
    for (size_t j = 1; j < vec.size(); ++j)
        if (end.x < vec[j].x || (end.x == vec[j].x && end.y < vec[j].y)) {
            end = vec[j];
        }

    return vec;
}

std::vector<Point> greckham_par(std::vector <Point> p, int vec_size) {
    MPI_Datatype POINT_TYPE;
    MPI_Type_contiguous(sizeof(int64_t) / sizeof(int) * 2, MPI_INT, &POINT_TYPE);
    MPI_Type_commit(&POINT_TYPE);

    int proc_size, proc;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &proc);

    int n = vec_size;
    int* send_array_counts = new int[proc_size];
    int* send_array_dist = new int[proc_size];

    if (proc == 0) {
        int length = 0;
        for (int i = 0; i < proc_size; i++) {
            send_array_counts[i] = (i < n% proc_size) ? (n / proc_size + 1) : (n / proc_size);
            send_array_dist[i] = length;
            length += (i < n% proc_size) ? (n / proc_size + 1) : (n / proc_size);
        }
    }

    int local_size;
    if (proc < n % proc_size)
        local_size = n / proc_size + 1;
    else
        local_size = n / proc_size;

    std::vector<Point> vector_local(local_size);
    MPI_Scatterv(p.data(), send_array_counts, send_array_dist, POINT_TYPE,
        vector_local.data(), local_size, POINT_TYPE, 0, MPI_COMM_WORLD);

    vector_local = greckham_seq(vector_local);
    local_size = vector_local.size();
    int left_process = proc_size, dim = 1;
    while (left_process > 1) {
        left_process = left_process / 2 + left_process % 2;
        if ((proc - dim) % (2 * dim) == 0) {
            MPI_Send(&local_size, 1, MPI_INT, proc - dim, 0, MPI_COMM_WORLD);
            MPI_Send(vector_local.data(), local_size, POINT_TYPE, proc - dim, 0, MPI_COMM_WORLD);
        }
        if ((proc % (2 * dim) == 0) && (proc_size - proc > dim)) {
            MPI_Status status;
            int recieved_size;

            MPI_Recv(&recieved_size, 1, MPI_INT, proc + dim, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            std::vector<Point> recieved_vector(recieved_size);
            MPI_Recv(recieved_vector.data(), recieved_size, POINT_TYPE, proc + dim, MPI_ANY_TAG,
                MPI_COMM_WORLD, &status);

            vector_local.insert(vector_local.end(), recieved_vector.begin(), recieved_vector.end());
            vector_local = greckham_seq(vector_local);
            local_size = vector_local.size();
        }
        dim = 2 * dim;
    }
    return vector_local;
}

void print_field(std::vector<Point> vec, int N) {
    bool f = false;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (auto& point : vec) {
                if (point.x == j && point.y == i) {
                    f = true;
                }
            }
            if (f) {
                std::cout << "o ";
                f = false;
            } else {
                std::cout << "` ";
            }
        }
        std::cout << "\n";
    }
}

std::vector <Point> get_field(int N, int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    std::vector <Point> vector;
    bool f = false;
    int x, y;
    for (int i = 0; i < size; i++) {
        x = gen() % N;
        y = gen() % N;
        for (auto& pt : vector) {
            if (pt.x == x && pt.y == y) {
                f = true;
            }
        }
        if (!f) {
            vector.push_back({ x, y });
        } else {
            --i;
        }
        f = false;
    }

    return vector;
}

int64_t rotation(Point a, Point b, Point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool distance(Point a, Point b, Point c) {
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) >=
    (a.x - c.x) * (a.x - c.x) + (a.y - c.y) * (a.y - c.y);
}
