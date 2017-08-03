#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>

bool isSame(unsigned int *a, unsigned int *b, int n)
{
	for (int i = 0; i < n; i++)
	{
		if (a[i] != b[i])
		{
			return false;
		}
	}

	return true;
}

void printResult(unsigned int *a, int n)
{
	for (int i = 0; i < n; i++)
	{
		std::cout << a[i] << " ";
	}

	std::cout << std::endl;
}

long long avg(std::vector<long long> v)
{
	return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

long long min(std::vector<long long> v)
{
	return *min_element(v.begin(), v.end());
}

long long max(std::vector<long long> v)
{
	return *max_element(v.begin(), v.end());
}