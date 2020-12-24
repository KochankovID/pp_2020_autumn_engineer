// Copyright 2020 Novikov Danil
#ifndef MODULES_TASK_3_NOVIKOV_D_JARVIS_NOVIKOV_D_JARVIS_H_
#define MODULES_TASK_3_NOVIKOV_D_JARVIS_NOVIKOV_D_JARVIS_H_

#include <vector>
#include <random>
#include <ctime>

struct point {
    int64_t x;
    int64_t y;

    point(int64_t x, int64_t y) : x(x), y(y) {}

    point() = default;

    friend bool operator==(point a, point b) {
        return b.x == a.x && b.y == a.y;
    }

    friend bool operator!=(point a, point b) {
        return !operator==(a, b);
    }
};

std::vector <point> get_random_image(int N, int size);

void print_image(std::vector<point> vec, int N);

std::vector<point> convex_hull_jarvis_sequential(std::vector <point> p);
std::vector<point> convex_hull_jarvis_parallel(std::vector <point> p, int vec_size);

#endif  // MODULES_TASK_3_NOVIKOV_D_JARVIS_NOVIKOV_D_JARVIS_H_
