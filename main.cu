
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <iostream>
#include <fstream>

#include "testing.cuh"

void printUsage(char *filename)
{
	std::cout << filename << " [filename] [start vertex number] [cpu tests count] [gpu tests count]" << std::endl;

	std::cout << "CSR format. File should contain in the first line number of vertices. Second line is offset row and third neighbours" << std::endl;
}

int main(int argc, char **argv)
{
	if (argc != 5) {
		printUsage(argv[0]);
		return -1;
	}

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		std::cout << "cudaSetDevice failed";
		return 1;
	}

	char *filename = argv[1];
	unsigned int startVetex = atoi(argv[2]);
	unsigned int cpuTests = atoi(argv[3]);
	unsigned int gpuTests = atoi(argv[4]);

	std::ifstream file;
	file.open(filename);

	if (!file.is_open())
	{
		std::cerr << "File not opened" << std::endl;
		return -1;
	}

	unsigned int nodes;
	file >> nodes;

	unsigned int *offsets = new unsigned int[nodes + 1];

	for (unsigned int i = 0; i < nodes + 1; i++)
	{
		file >> offsets[i];
	}

	unsigned int *neigbours = new unsigned int[offsets[nodes]];

	for (unsigned int i = 0; i < offsets[nodes]; i++)
	{
		file >> neigbours[i];
	}

	runTests(offsets, neigbours, nodes, startVetex, cpuTests, gpuTests);

	delete[] offsets;
	delete[] neigbours;

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
		std::cout << "cudaDeviceReset failed!";
        return 1;
    }

    return 0;
}