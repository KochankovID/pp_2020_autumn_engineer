// Copyright 2020 Kamskov Evgeny
#ifndef MODULES_TASK_3_KAMSKOV_E_GRACKHAM_HULL_KAMSKOV_E_GRACKHAM_HULL_H_
#define MODULES_TASK_3_KAMSKOV_E_GRACKHAM_HULL_KAMSKOV_E_GRACKHAM_HULL_H_

#include <vector>
#include <random>
#include <ctime>

struct Point {
    int64_t x;
    int64_t y;

    Point(int64_t x, int64_t y) : x(x), y(y) {}

    Point() = default;

    friend bool operator==(Point a, Point b) {
        return b.x == a.x && b.y == a.y;
    }

    friend bool operator!=(Point a, Point b) {
        return !operator==(a, b);
    }
};

int64_t rotation(Point a, Point b, Point c);
bool distance(Point a, Point b, Point c);

std::vector<Point> get_field(int N, int size);

void print_field(std::vector<Point> vec, int N);

std::vector<Point> greckham_seq(std::vector < Point > p);
std::vector<Point> greckham_par(std::vector <Point> p, int vec_size);

#endif  // MODULES_TASK_3_KAMSKOV_E_GRACKHAM_HULL_KAMSKOV_E_GRACKHAM_HULL_H_
