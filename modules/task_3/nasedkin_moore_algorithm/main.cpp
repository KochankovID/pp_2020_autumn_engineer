// Copyright 2020 Nasedkin Nikita
#include <gtest-mpi-listener.hpp>
#include <gtest/gtest.h>
#include <vector>
#include "./moore_algotihtm.h"
// mpiexec -n 4 "C:\Labs\MPI2020\pp_2020_autumn_engineer\build\bin\nasedkin_moore_algorithm_mpi.exe"
TEST(Parallel_Operations_MPI, Can_Make_Random_Graph) {
  std::vector<int> Graph;
  ASSERT_NO_THROW(Graph = GetRandomGraph(50));
}

TEST(Parallel_Operations_MPI, Can_Make_Sequential_Calculating) {
  std::vector<int> Graph;
  Graph = GetRandomGraph(50);
  ASSERT_NO_THROW(MooreSequential(&Graph, 0));
}

TEST(Parallel_Operations_MPI, Can_Make_Parallel_Calculating) {
  std::vector<int> Graph;
  Graph = GetRandomGraph(50);
  ASSERT_NO_THROW(MooreParallel(&Graph, 0));
}

TEST(Parallel_Operations_MPI, Is_Sequential_EQ_Parallel) {
  int size = 10;
  std::vector<int> Graph;
  std::vector<int> sequentialAns;
  std::vector<int> parallelAns;
  Graph = GetRandomGraph(size);
  sequentialAns = MooreSequential(&Graph, 0);
  parallelAns = MooreParallel(&Graph, 0);
  for (int i = 0; i < size; i++) {
    EXPECT_EQ(sequentialAns[i], parallelAns[i]);
  }
}

TEST(Parallel_Operations_MPI, Is_Parallel_Faster_Then_Sequential) {
  int size = 20;
  int proc_rank;
  std::vector<int> graph;
  std::vector<int> result;
  MPI_Comm_rank(MPI_COMM_WORLD, &proc_rank);
  double start_time, end_time, parallel_time, sequential_time;
  graph = GetRandomGraph(size);

  start_time = MPI_Wtime();
  result = MooreParallel(&graph, 0);
  end_time = MPI_Wtime();
  if (proc_rank == 0) {
    parallel_time = end_time - start_time;
  }

  if (proc_rank == 0) {
    start_time = MPI_Wtime();
    result = MooreSequential(&graph, 0);
    end_time = MPI_Wtime();
    sequential_time = end_time - start_time;

    std::cout << "sequential_time: " << sequential_time << std::endl
      << "parallel_time: " << parallel_time << std::endl;
  }
}


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  MPI_Init(&argc, &argv);
  ::testing::AddGlobalTestEnvironment(new GTestMPIListener::MPIEnvironment);
  ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
  listeners.Release(listeners.default_result_printer());
  listeners.Release(listeners.default_xml_generator());
  listeners.Append(new GTestMPIListener::MPIMinimalistPrinter);
  return RUN_ALL_TESTS();
}
