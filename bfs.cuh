#pragma once

int bfsWithGpu(const unsigned int *offsets, const unsigned int *neighbours, const unsigned int vertCount, const unsigned int startVert, unsigned int *distances);
int bfsWithCpu(const unsigned int *offsets, const unsigned int *neighbours, const unsigned int vertCount, const unsigned int startVert, unsigned int *distances);