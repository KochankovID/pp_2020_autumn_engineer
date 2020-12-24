// Copyright 2018 Nesterov Alexander
#include "../../../modules/task_3/frolova_o_radix_sort_simple_merge_update/radix_sort_simple_merge.h"
#include <mpi.h>
#include <math.h>
#include <iostream>
#include <string>
#include <random>
#include <ctime>
#include <algorithm>
#include <vector>

std::vector<int> getRandomVector(int size) {
  std::vector<int> vec(size);
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  for (int i = 0; i < size; i++)
     vec[i] = gen() % 10000;
  return vec;
}

std::vector<int> checkVector(std::vector<int> vec) {
  int j = 0;
  int tmp = 0;
  int lengh = vec.size();
  for (int i = 0; i < lengh; i++) {
    j = i;
    for (int k = i; k < lengh; k++) {
        if (vec[j] > vec[k]) {
           j = k;
        }
    }
    tmp = vec[i];
    vec[i] = vec[j];
    vec[j] = tmp;
  }
  return vec;
}

int countRadix(int num) {
  int res = 0;
  while (num > 0) {
        num = num / 10;
        res++;
  }
  return res;
}

int maxVal(std::vector<int> vec) {
  int size = vec.size();
  int res = 0;
  for (int i = 0; i < size - 1; i++) {
     if (vec[i] < vec[i + 1]) res = vec[i + 1];
  }
  return res;
}

std::vector<int> merge(std::vector<int> vec1, std::vector<int> vec2) {
  int size1 = vec1.size();
  int size2 = vec2.size();
  std::vector<int> res((size1 + size2));
  int i, j, k;
  i = j = k = 0;
  while (i < size1 && j < size2) {
        if (vec1[i] < vec2[j])
           res[k] = vec1[i++];
        else
           res[k] = vec2[j++];
        k++;
  }
  while (i < size1) {
        res[k++] = vec1[i++];
  }
  while (j < size2) {
        res[k++] = vec2[j++];
  }
  return res;
}

std::vector<int> radixSort(std::vector<int> vec, int radix) {
  std::vector<int> counts(10);
  std::vector<int> counts1(10);
  std::vector<int> res(vec.size());
  int tmp = 0;
  int size = vec.size();
  for (int i = 0; i < 10; i++)
      counts[i] = counts1[i] = 0;
  for (int i = 0; i < size; i++)
      counts[((vec[i] % static_cast<unsigned int>(pow(10, (radix)))) / pow(10, (radix - 1)))]++;
  for (int i = 1; i < 10; i++)
      for (int j = i - 1; j >= 0; j--)
          counts1[i] += counts[j];
  for (int i=0; i < size; i++) {
      tmp = vec[i];
      int index = (tmp % static_cast<unsigned int>(pow(10, (radix)))) / pow(10, (radix - 1));
      res[counts1[index]] = tmp;
      counts1[index]++;
  }
  return res;
}

std::vector<int> ParallelRadixSort(std::vector<int> vec, int size1) {
  int radix = countRadix(maxVal(vec));
  int size, rank;
  int size_vec = size1;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int delta = size_vec / size;
  MPI_Barrier(MPI_COMM_WORLD);
  std::vector<int> local_vec(delta);
  if (rank == 0) {
     for (int i = 1; i < size; i++) {
         if ((i == (size - 1))&&((size_vec % size) != 0)) {
            int delta_for_last = size_vec - (size - 1) * delta;
            MPI_Send(&vec[0] + i * delta, delta_for_last, MPI_INT, i, 0, MPI_COMM_WORLD);
         } else {
            MPI_Send(&vec[0] + i * delta, delta, MPI_INT, i, 0, MPI_COMM_WORLD);
         }
     }
  }
  if (rank == 0) {
     local_vec = std::vector<int>(vec.begin(), vec.begin() + delta);
  } else {
      MPI_Status status;
      if ((rank == (size - 1))&&((size_vec % size) != 0)) {
         delta = size_vec - (size - 1)*delta;
         local_vec.resize(delta);
      }
      MPI_Recv(&local_vec[0], delta, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) {
     for (int i = 1; i < size; i++) {
        MPI_Send(&radix, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
     }
  } else {
        MPI_Status status;
        MPI_Recv(&radix, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    }
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 1; i <= radix; i++) {
      local_vec = radixSort(local_vec, i);
  }
  MPI_Barrier(MPI_COMM_WORLD);
  int index = (rank + size) % size + 1;
  MPI_Barrier(MPI_COMM_WORLD);
  std::vector<int> res(0);
  MPI_Status status;
  if (rank == 0) {
     int size_recive_vec1 = 0;
     int size_recive_vec2 = 0;
     std::vector<int> recive_vec1;
     std::vector<int> recive_vec2;
     if (2 * index <= size) {
        MPI_Recv(&size_recive_vec1, 1, MPI_INT, ((index * 2 - 1) % size), 0, MPI_COMM_WORLD, &status);
        recive_vec1.resize(size_recive_vec1);
        MPI_Recv(&recive_vec1[0], size_recive_vec1, MPI_INT, ((index * 2 - 1) % size), 0, MPI_COMM_WORLD, &status);
     }
     if (2 * index + 1  <= size) {
        MPI_Recv(&size_recive_vec2, 1, MPI_INT, ((index * 2) % size), 0, MPI_COMM_WORLD, &status);
        recive_vec2.resize(size_recive_vec2);
        MPI_Recv(&recive_vec2[0], size_recive_vec2, MPI_INT, ((index * 2) % size), 0, MPI_COMM_WORLD, &status);
     }
     if (!recive_vec1.empty() && !recive_vec2.empty()) {
        res = merge(recive_vec1, recive_vec2);
        res = merge(res, local_vec);
     } else if (!recive_vec1.empty()) {
         res = merge(recive_vec1, local_vec);
     } else if (!recive_vec2.empty()) {
         res = merge(recive_vec2, local_vec);
     } else {
         res = local_vec;
     }
  } else {
      std::vector<int> recive_vec1;
      std::vector<int> recive_vec2;
      int size_recive_vec1 = 0;
      int size_recive_vec2 = 0;
      if (2 * index <= size) {
         MPI_Recv(&size_recive_vec1, 1, MPI_INT, ((index * 2 - 1) % size), 0, MPI_COMM_WORLD, &status);
         recive_vec1.resize(size_recive_vec1);
         MPI_Recv(&recive_vec1[0], size_recive_vec1, MPI_INT, ((index * 2 - 1) % size), 0, MPI_COMM_WORLD, &status);
      }
      if (2 * index + 1 <= size) {
         MPI_Recv(&size_recive_vec2, 1, MPI_INT, ((index * 2) % size), 0, MPI_COMM_WORLD, &status);
         recive_vec2.resize(size_recive_vec2);
         MPI_Recv(&recive_vec2[0], size_recive_vec2, MPI_INT, ((index * 2) % size), 0, MPI_COMM_WORLD, &status);
      }
  std::vector<int> send_vec;
  if (!recive_vec1.empty() && !recive_vec2.empty()) {
     send_vec = merge(recive_vec1, recive_vec2);
     send_vec = merge(send_vec, local_vec);
  } else if (!recive_vec1.empty()) {
      send_vec = merge(recive_vec1, local_vec);
  } else if (!recive_vec2.empty()) {
      send_vec = merge(recive_vec2, local_vec);
  } else {
      send_vec = local_vec;
  }
    int size_cur_vec = send_vec.size();
    MPI_Send(&size_cur_vec, 1, MPI_INT, ((index / 2 - 1) % size), 0, MPI_COMM_WORLD);
    MPI_Send(&send_vec[0], send_vec.size(), MPI_INT, ((index / 2 - 1) % size), 0, MPI_COMM_WORLD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    return res;
}

void difOfTime(int size1) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double timeRadixSort1, timeRadixSort2, timeSort1, timeSort2;
  std::vector<int> vec;
  if (rank == 0) vec = getRandomVector(size1);
  if (rank == 0)
     timeRadixSort1 = MPI_Wtime();
  ParallelRadixSort(vec, size1);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
     timeRadixSort2 = MPI_Wtime();
  if (rank == 0) {
     timeSort1 = MPI_Wtime();
     int radix = countRadix(maxVal(vec));
     for (int i = 1; i <= radix; i++)
        vec = radixSort(vec, radix);
     timeSort2 = MPI_Wtime();
     double resRadixSort = timeRadixSort2 - timeRadixSort1;
     double resSort = timeSort2 - timeSort1;
     if (resSort < resRadixSort) {
        std::cout << "SimpleSort faster than ParallelRadixSort on " << resRadixSort - resSort << "\n";
     } else {
        std::cout << "ParallelRadixSort faster than SimpleSort on " << resSort / resRadixSort << "\n";
    }
  }
}

void timeOfRadixSort(int size1) {
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  double timeRadixSort1, timeRadixSort2;
  std::vector<int> vec;
  if (rank == 0) {
     vec = getRandomVector(size1);
     timeRadixSort1 = MPI_Wtime();
     }
  MPI_Barrier(MPI_COMM_WORLD);
  ParallelRadixSort(vec, size1);
  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0)
     timeRadixSort2 = MPI_Wtime();
  if (rank == 0) {
     double resRadixSort = timeRadixSort2 - timeRadixSort1;
     std::cout << "time is  " << resRadixSort << "\n";
  }
}
