#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define gpuErrchk(ans) if (ans != cudaSuccess) {std::cerr <<  "GPUassert: " <<  cudaGetErrorString(ans) << " " << __FILE__ << " " << __LINE__ << std::endl; goto Error;}

__global__ void bfsKernel(const unsigned int *offsets, const unsigned int *neighbours, const unsigned int vertCount, const unsigned int startVert, unsigned int *distances, unsigned int *currentQueue, unsigned int *nextQueue)
{
	int idx = threadIdx.x;

	for (int k = idx; k < vertCount; k += blockDim.x)
	{
		if (k == startVert)
		{
			distances[k] = 0;
		}
		else
		{
			distances[k] = UINT_MAX;
		}
	}

	__shared__ unsigned int currentQueueLength;
	__shared__ unsigned int nextQueueLength;

	if (idx == 0)
	{
		currentQueue[0] = startVert;
		currentQueueLength = 1;
		nextQueueLength = 0;
	}

	__syncthreads();

	while (1)
	{
		for (int queueIdx = idx; queueIdx < currentQueueLength; queueIdx += blockDim.x)
		{
			unsigned int vertex = currentQueue[queueIdx];

			for (unsigned int n = offsets[vertex]; n < offsets[vertex + 1]; n++)
			{
				unsigned int neighbourVertex = neighbours[n];

				if (atomicCAS(&distances[neighbourVertex], UINT_MAX, distances[vertex] + 1) == UINT_MAX)
				{
					int t = atomicAdd(&nextQueueLength, 1);
					nextQueue[t] = neighbourVertex;
				}
			}
		}

		__syncthreads();

		if (nextQueueLength == 0)
		{
			break;
		}
		else
		{
			for (int queueIdx = idx; queueIdx < nextQueueLength; queueIdx += blockDim.x)
			{
				currentQueue[queueIdx] = nextQueue[queueIdx];
			}

			__syncthreads();

			if (idx == 0)
			{
				currentQueueLength = nextQueueLength;
				nextQueueLength = 0;
			}
			__syncthreads();
		}
	}
}

int bfsWithGpu(const unsigned int *offsets, const unsigned int *neighbours, const unsigned int vertCount, const unsigned int startVert, unsigned int *distances)
{
	unsigned int *dev_offsets = 0;
	unsigned int *dev_neigbours = 0;
	unsigned int *dev_distances = 0;

	unsigned int *dev_currentQueue = 0;
	unsigned int *dev_nextQueue = 0;

	const unsigned int edgesCount = offsets[vertCount];

	gpuErrchk(cudaMalloc((void**)&dev_offsets, (vertCount + 1) * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_neigbours, (edgesCount) * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_distances, (vertCount) * sizeof(unsigned int)));

	gpuErrchk(cudaMalloc((void**)&dev_currentQueue, (vertCount) * sizeof(unsigned int)));
	gpuErrchk(cudaMalloc((void**)&dev_nextQueue, (vertCount) * sizeof(unsigned int)));

	gpuErrchk(cudaMemcpy(dev_offsets, offsets, (vertCount + 1) * sizeof(unsigned int), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(dev_neigbours, neighbours, (edgesCount) * sizeof(unsigned int), cudaMemcpyHostToDevice));

	std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();
	bfsKernel<<<1, 1024>>>(dev_offsets, dev_neigbours, vertCount, startVert, dev_distances, dev_currentQueue, dev_nextQueue);

	gpuErrchk(cudaGetLastError());
	gpuErrchk(cudaDeviceSynchronize());

	std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();

	std::cout << "Result (kernel): " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << std::endl;

	gpuErrchk(cudaMemcpy(distances, dev_distances, (vertCount) * sizeof(unsigned int), cudaMemcpyDeviceToHost));

Error:
	cudaFree(dev_offsets);
	cudaFree(dev_neigbours);
	cudaFree(dev_distances);

	cudaFree(dev_currentQueue);
	cudaFree(dev_nextQueue);

	return 0;
}