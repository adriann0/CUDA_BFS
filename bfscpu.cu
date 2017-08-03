#include <climits>
#include <queue>

#include "bfs.cuh"

int bfsWithCpu(const unsigned int *offsets, const unsigned int *neighbours, const unsigned int vertCount, const unsigned int startVert, unsigned int *distances)
{
	for (unsigned int i = 0; i < vertCount; i++) {
		distances[i] = UINT_MAX;
	}

	distances[startVert] = 0;

	std::queue<unsigned int> frontiers;
	frontiers.push(startVert);

	while (!frontiers.empty())
	{
		unsigned int v = frontiers.front();
		frontiers.pop();

		for (unsigned int n = offsets[v]; n < offsets[v + 1]; n++) {
			unsigned int neighbour = neighbours[n];

			if (distances[neighbour] == UINT_MAX)
			{
				distances[neighbour] = distances[v] + 1;
				frontiers.push(neighbour);
			}
		}
	}

	return 0;
}