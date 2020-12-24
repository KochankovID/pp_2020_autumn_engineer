// Copyright 2018 Nesterov Alexander
#include <vector>
#include "../../modules/task_3/ptitsyn_p_jarvis/jarvis.h"

std::vector<Point> createRandomPoints(size_t size) {
    std::vector<Point> result(size);
    std::random_device rd;
    std::mt19937 mersenne(rd());
    std::uniform_real_distribution<> urd(-10000.0, 10000.0);
    for (size_t i = 0; i < size; i++) {
        result[i].x = static_cast<int>(urd(mersenne));
        result[i].y = static_cast<int>(urd(mersenne));
    }
    return result;
}

int orientation(const Point p, const Point q, const Point r) {
    double val;
    val = (q.y - p.y) * (r.x - q.x) -
        (q.x - p.x) * (r.y - q.y);
    if (val == 0) return 0;
    return (val > 0) ? 1 : 2;
}

size_t minpoint(const std::vector<Point>& points) {
    int l = 0;
    for (int i = 1; i < static_cast<int>(points.size()); i++)
        if (points[i].x < points[l].x)
            l = i;
    return l;
}

int searchMinPointParallel(const std::vector<Point>& points, int rank, int size) {
    int l = minpoint(points);
    std::vector<Point> local_points(size);
    MPI_Gather(&l, 1, MPI_INT, &local_points[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        l = minpoint(local_points);
    }
    MPI_Bcast(&l, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return l;
}

std::vector<Point> convexHull(const std::vector<Point>& points) {
    if (points.size() < 3)
        return points;
    std::vector<Point> hull;
    int l = minpoint(points);
    int p = l, q;
    do {
        hull.push_back(points[p]);
        q = (p + 1) % points.size();
        for (int i = 0; i < static_cast<int>(points.size()); i++) {
            if (orientation(points[p], points[i], points[q]) == 2)
                q = i;
        }
        p = q;
    } while (p != l);
    return hull;
}

std::vector<Point> convexHullPar(const std::vector<Point>& points) {
    int size, rank;
    double startT;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);;
    if (rank == 0)
        startT = MPI_Wtime();
    if (points.size() < 3)
        return points;
    if (size == 1)
        return convexHull(points);
    int elements_count = static_cast<int>(points.size());
    int delta = elements_count / size;
    int ost = elements_count % size;
    if ((delta == 1) || (static_cast<size_t>(size) > points.size())) {
        if (rank == 0) {
            return convexHull(points);
        } else {
            return points;
        }
    }
    int* revcount = new int[size];
    int* displs = new int[size];
    revcount[0] = 2 * (delta + ost);
    displs[0] = 0;
    for (int i = 1; i < size; i++) {
        revcount[i] = delta * 2;
        displs[i] = (delta * i + ost) * 2;
    }
    std::vector<Point> part_vector(rank == 0 ? delta + ost : delta);
    MPI_Scatterv(&points[0], revcount, displs, MPI_INT, &part_vector[0], revcount[rank], MPI_INT, 0, MPI_COMM_WORLD);
    int l = searchMinPointParallel(part_vector, rank, size);
    std::vector<Point> hull;
    std::vector<Point> local_points(size);
    int p = l, q;
    do {
        hull.push_back(points[p]);
        q = (p + 1) % points.size();
        for (int i = 0; i < static_cast<int>(points.size()); i++) {
            if (orientation(points[p], points[i], points[q]) == 2)
                q = i;
        }
        p = q;
        MPI_Gather(&q, 1, MPI_INT, &local_points[0], 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            for (int i = 0; i < static_cast<int>(local_points.size()); i++) {
                if (orientation(points[p], local_points[i], points[q]) == 2)
                    q = i;
            }
        }
        if (hull.size() == points.size()) {
            break;
        }
    } while (p != l);
    if (rank == 0) {
        double endT = MPI_Wtime();
        std::cout << "MPI time: " << endT - startT << std::endl;
    }
    return hull;
}
