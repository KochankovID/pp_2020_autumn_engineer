// Copyright 2020 Bulychev Andrey
#include "../../../modules/task_3/bulychev_a_radix_sort/radix_sort.h"

#include <math.h>
#include <mpi.h>

#include <algorithm>
#include <ctime>
#include <iostream>
#include <random>
#include <string>
#include <vector>

std::vector<int> getRandomVector(int size) {
  std::vector<int> vec(size);
  std::mt19937 gen;
  gen.seed(static_cast<unsigned int>(time(0)));
  for (int i = 0; i < size; i++) vec[i] = gen() % 10000;
  return vec;
}

int pow10(int power) {
  int tmp = 1;
  for (int i = 0; i < power; i++) {
    tmp = tmp * 10;
  }
  return tmp;
}

std::vector<int> bubble_sort_Vector(std::vector<int> vec) {
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

int pow2(int pow) {
  int res = 1;
  for (int i = 0; i < pow; ++i) res = res * 2;

  return res;
}

bool vector_is_sort(std::vector<int> vec) {
  int size = vec.size();
  for (int i = 0; i < size - 1; i++) {
    if (vec[i] > vec[i + 1]) {
      return false;
    }
  }
  return true;
}

int get_max_power(std::vector<int> vec) {
  int max_power = 0;
  int tmp = vec[0];
  int size = vec.size();
  for (int i = 0; i < size; i++) {
    if (tmp < vec[i]) {
      tmp = vec[i];
    }
  }
  if (tmp == 0) {
    return 1;
  }
  while (tmp != 0) {
    tmp = tmp / 10;
    max_power++;
  }
  return max_power;
}

std::vector<int> RadixSort(std::vector<int> vec, int size) {
  std::vector<std::vector<int>> main_vec(10);
  std::vector<int> my_vec = vec;
  int tmp = 0;
  int power = 1;
  int max_power;
  if (vector_is_sort(vec)) {
    return vec;
  }
  max_power = get_max_power(vec);
  while (power <= max_power) {
    for (int i = 0; i < size; i++) {
      tmp = (my_vec[i] % static_cast<unsigned int>(pow10(power))) /
            static_cast<unsigned int>(pow(10, power - 1));
      main_vec[tmp].push_back(my_vec[i]);
    }
    my_vec.clear();
    for (int i = 0; i < 10; i++) {
      if (!main_vec[i].empty()) {
        int size = main_vec[i].size();
        for (int j = 0; j < size; j++) {
          my_vec.push_back(main_vec[i][j]);
        }
        main_vec[i].clear();
      }
    }
    power++;
  }
  return my_vec;
}

std::vector<int> Merge(std::vector<int> vec1, std::vector<int> vec2) {
  int size1 = vec1.size();
  int size2 = vec2.size();
  std::vector<int> tmp(size1 + size2);
  int i = 0, j = 0;
  for (int k = 0; k < size1 + size2; k++) {
    if (i > size1 - 1) {
      int a = vec2[j];
      tmp[k] = a;
      j++;
    } else {
      if (j > size2 - 1) {
        int a = vec1[i];
        tmp[k] = a;
        i++;
      } else {
        if (vec1[i] < vec2[j]) {
          int a = vec1[i];
          tmp[k] = a;
          i++;
        } else {
          int b = vec2[j];
          tmp[k] = b;
          j++;
        }
      }
    }
  }
  return tmp;
}

std::vector<int> ParallSort(std::vector<int> src) {
  int mpi_rank, mpi_size;
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  int length, rem;
  if (mpi_rank == 0) {
    length = src.size() / mpi_size;
    rem = src.size() % mpi_size;
  }
  MPI_Bcast(&length, 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<int> local_vec(length);
  MPI_Scatter(src.data(), length, MPI_INT, local_vec.data(), length, MPI_INT, 0,
              MPI_COMM_WORLD);
  if (mpi_rank == 0 && rem != 0) {
    local_vec.insert(local_vec.end(), src.end() - rem, src.end());
  }
  local_vec = RadixSort(local_vec, local_vec.size());
  int i = mpi_size;
  int num_op = 1;
  int num_op_length = length;
  int part;
  int loc_size = length;
  if (mpi_rank == 0) {
    loc_size = loc_size + rem;
  }
  while (i > 1) {
    if (i % 2 == 1) {
      if (mpi_rank == 0) {
        MPI_Status status1;
        std::vector<int> tmp_vec(num_op_length);
        part = pow2(num_op - 1) * (i - 1);
        MPI_Recv(tmp_vec.data(), num_op_length, MPI_INT, part, 1,
                 MPI_COMM_WORLD, &status1);
        local_vec = Merge(local_vec, tmp_vec);
        loc_size += num_op_length;
      }
      if (mpi_rank == pow2(num_op - 1) * (i - 1)) {
        MPI_Send(local_vec.data(), num_op_length, MPI_INT, 0, 1,
                 MPI_COMM_WORLD);
        return local_vec;
      }
    }
    if (mpi_rank % pow2(num_op) == 0) {
      part = mpi_rank + pow2(num_op - 1);
      std::vector<int> tmp_vec(num_op_length);
      MPI_Status status3;
      MPI_Recv(tmp_vec.data(), num_op_length, MPI_INT, part, 3, MPI_COMM_WORLD,
               &status3);
      local_vec = Merge(local_vec, tmp_vec);
      loc_size += num_op_length;
    }
    if (mpi_rank % pow2(num_op) != 0) {
      part = mpi_rank - pow2(num_op - 1);
      MPI_Send(local_vec.data(), num_op_length, MPI_INT, part, 3,
               MPI_COMM_WORLD);
      return local_vec;
    }
    num_op++;
    i = i / 2;
    num_op_length = num_op_length * 2;
  }
  return local_vec;
}
