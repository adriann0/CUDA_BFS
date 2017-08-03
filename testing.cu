#include <iostream>
#include <chrono>
#include <vector>

#include "bfs.cuh"
#include "utils.cuh"

long long runOnce(int (*fun)(const unsigned int *offsets, const unsigned int *neighbours, const unsigned int vertCount, const unsigned int startVert, unsigned int *distances), const unsigned int *offsets, const unsigned int *neighbours, const unsigned int vertCount, const unsigned int startVert, unsigned int *results)
{
	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	int result = fun(offsets, neighbours, vertCount, startVert, results);
	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	if (result)
	{
		std::cout << "Error during computation occured\n";
		exit(result);
	}

	return std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
}

void runTests(const unsigned int *offsets, const unsigned int *neighbours, const unsigned int vertCount, const unsigned int startVert, int cpuRep, int gpuRep)
{
	unsigned int *resultsCpu = nullptr;
	unsigned int *resultsGpu = nullptr;
	
	std::vector<long long> timeResultsCpu;
	std::vector<long long> timeResultsGpu;

	for (int i = 1; i <= cpuRep; i++) 
	{
		std::cout << "Running CPU test " << i << "..." << std::endl;
		delete[] resultsCpu;
		resultsCpu = new unsigned int[vertCount];
		long long timeResult = runOnce(bfsWithCpu, offsets, neighbours, vertCount, startVert, resultsCpu);
		std::cout << "Result (overall): " << timeResult << std::endl;
		timeResultsCpu.push_back(timeResult);
	}

	std::cout << "-----------------------" << std::endl;

	for (int i = 1; i <= gpuRep; i++)
	{
		std::cout << "Running GPU test " << i << "..." << std::endl;
		delete[] resultsGpu;
		resultsGpu = new unsigned int[vertCount];
		long long timeResult = runOnce(bfsWithGpu, offsets, neighbours, vertCount, startVert, resultsGpu);
		std::cout << "Result (overall): " << timeResult << std::endl;
		timeResultsGpu.push_back(timeResult);
	}

	std::cout << std::endl;
	std::cout << (isSame(resultsCpu, resultsGpu, vertCount) ? "Results OK" : "RESULTS NOT THE SAME!!!");
	std::cout << std::endl;

	delete[] resultsCpu;
	delete[] resultsGpu;
}