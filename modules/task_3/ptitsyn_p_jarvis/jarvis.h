// Copyright 2018 Nesterov Alexander
#ifndef MODULES_TASK_3_PTITSYN_P_JARVIS_JARVIS_H_
#define MODULES_TASK_3_PTITSYN_P_JARVIS_JARVIS_H_
#include <mpi.h>
#include <vector>
#include <iostream>
#include <ctime>
#include <random>

struct Point {
    int x, y;
    Point() = default;
    Point(int X, int Y) {
        x = X;
        y = Y;
    }

    const Point operator=(const Point& point) {
        x = point.x;
        y = point.y;
        return *this;
    }

    bool operator!=(const Point& point) {
        return ((x != point.x) || (y != point.y));
    }

    bool operator==(const Point& point) {
        return ((x == point.x) && (y == point.y));
    }
};
int searchMinPointParallel(const std::vector<Point>& points, int rank, int size);
std::vector<Point> convexHullPar(const std::vector<Point>& points);
size_t minpoint(const std::vector<Point>& points);
std::vector<Point> createRandomPoints(size_t size);
std::vector<Point> convexHull(const std::vector<Point>& points);
int orientation(const Point p, const Point q, const Point r);
#endif  // MODULES_TASK_3_PTITSYN_P_JARVIS_JARVIS_H_
