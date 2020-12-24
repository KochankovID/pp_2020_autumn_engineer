// Copyright 2020 Novikov Danil
#include <mpi.h>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <utility>
#include <vector>
#include <cstring>
#include "../../../modules/task_3/novikov_d_jarvis/novikov_d_jarvis.h"

void print_image(std::vector<point> vec, int N) {
    bool flag = false;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (auto& pt : vec) {
                if (pt.x == j && pt.y == i) {
                    flag = true;
                }
            }
            if (flag) {
                std::cout << "o ";
                flag = false;
            } else {
                std::cout << "- ";
            }
        }
        std::cout << "\n";
    }
}

std::vector <point> get_random_image(int N, int size) {
    std::mt19937 gen;
    gen.seed(static_cast<unsigned int>(time(0)));

    std::vector <point> result;
    bool flag = false;
    int64_t x, y;
    for (int i = 0; i < size; i++) {
        x = gen() % N;
        y = gen() % N;
        for (auto& pt : result) {
            if (pt.x == x && pt.y == y) {
                flag = true;
            }
        }
        if (!flag) {
            result.push_back({ x, y });
        } else {
            --i;
        }
        flag = false;
    }

    return result;
}

int64_t area_triangle(point a, point b, point c) {
    return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

bool point_in_box(point a, point b, point c) {
    return (a.x - b.x)*(a.x - b.x) + (a.y - b.y)*(a.y - b.y) >= (a.x - c.x)*(a.x - c.x) + (a.y - c.y)*(a.y - c.y);
}

std::vector<point> convex_hull_jarvis_sequential(std::vector <point> p) {
    std::vector <int> ip;
    int n = p.size();
    int first, q, next, i;
    int64_t sign;
    first = 0;
    for (i = 1; i < n; ++i)
        if (p[i].x < p[first].x || (p[i].x == p[first].x && p[i].y < p[first].y))
            first = i;

    q = first;
    do {
        ip.push_back(q);
        next = q;
        for (i = n - 1; i >= 0; --i)
            if (p[i] != p[q] && p[i] != p[next]) {
                sign = area_triangle(p[q], p[i], p[next]);
                if (next == q || sign < 0 || (sign == 0 && point_in_box(p[q], p[i], p[next])))
                    next = i;
            }
        q = next;
        if (ip.size() > p.size()) {
            for (size_t i = 0; i < p.size(); i++) {
                std::cout << p[i].x << " , " << p[i].y << std::endl;
            }
            std::cout << "infinite loop error" << std::endl;
            exit(-1);
        }
    } while (p[q] != p[first]);

    std::vector<point> res(ip.size());
    for (size_t i = 0; i < ip.size(); i++) {
        res[i] = p[ip[i]];
    }

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return res;
}

std::vector<point> convex_hull_jarvis_parallel(std::vector <point> p, int vec_size) {
    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Datatype dt_point;
    MPI_Type_contiguous(2 * sizeof(int64_t) / sizeof(int), MPI_INT, &dt_point);
    MPI_Type_commit(&dt_point);

    int n = vec_size;
    int* send_counts = new int[size];
    int* send_dist = new int[size];

    if (rank == 0) {
        int dif = 0;
        for (int i = 0; i < size; i++) {
            send_counts[i] = (i < n% size) ? (n / size + 1) : (n / size);
            send_dist[i] = dif;
            dif += (i < n% size) ? (n / size + 1) : (n / size);
        }
    }

    int k;
    if (rank < n % size)
        k = n / size + 1;
    else
        k = n / size;

    std::vector<point> local_buf(k);
    std::vector <int> ip;
    MPI_Scatterv(p.data(), send_counts, send_dist, dt_point, local_buf.data(), k, dt_point, 0, MPI_COMM_WORLD);

    local_buf = convex_hull_jarvis_sequential(local_buf);
    k = local_buf.size();

    int s = size, m = 1;
    while (s > 1) {
        s = s / 2 + s % 2;
        if ((rank - m) % (2 * m) == 0) {
            MPI_Send(&k, 1, MPI_INT,
                rank - m, 0, MPI_COMM_WORLD);
            MPI_Send(local_buf.data(), k, dt_point,
                rank - m, 0, MPI_COMM_WORLD);
        }
        if ((rank % (2 * m) == 0) && (size - rank > m)) {
            MPI_Status status;
            int k1;
            MPI_Recv(&k1, 1, MPI_INT,
                rank + m, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            std::vector<point> local_buf_2(k1);
            MPI_Recv(local_buf_2.data(), k1, dt_point,
                rank + m, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
            local_buf.insert(local_buf.end(), local_buf_2.begin(), local_buf_2.end());
            local_buf = convex_hull_jarvis_sequential(local_buf);
            k = local_buf.size();
        }
        m = 2 * m;
    }
    return local_buf;
}
