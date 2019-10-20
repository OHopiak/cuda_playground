#include "gpu_floyd.h"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"

__global__ void wake_gpu(int reps)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= reps)return;
}

void gpu_floyd(int* graph, int* paths, uint size)
{
	wake_gpu<<<1, BLOCK_SIZE>>> (32);
	//allocate device memory and copy graph data from host
	int* dG, * dP;
	size_t numBytes = size * size * sizeof(int);
	cudaError_t err = cudaMalloc(&dG, numBytes);
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
	err = cudaMalloc(&dP, numBytes);
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
	//copy from host to device graph info
	err = cudaMemcpy(dG, graph, numBytes, _HTD);
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
	err = cudaMemcpy(dP, paths, numBytes, _HTD);
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }

	dim3 dimGrid((size + BLOCK_SIZE - 1) / BLOCK_SIZE, size);

	for (int k = 0; k < size; k++) {//main loop

		gpu_floyd_kernel <<< dimGrid, BLOCK_SIZE >>> (k, dG, dP, size);
		err = ::cudaDeviceSynchronize();
		if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
	}
	//copy back memory
	err = cudaMemcpy(graph, dG, numBytes, _DTH);
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
	err = cudaMemcpy(paths, dP, numBytes, _DTH);
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }

	//free device memory
	err = cudaFree(dG);
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
	err = cudaFree(dP);
	if (err != cudaSuccess) { printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__); }
}

__global__ void gpu_floyd_kernel(int k, int* adjacency_mtx, int* paths, int size)
{
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (col >= size)return;
	int idx = size * blockIdx.y + col;

	__shared__ int best;
	if (threadIdx.x == 0)
		best = adjacency_mtx[size * blockIdx.y + k];
	__syncthreads();
	if (best == INF)
		return;
	int tmp_b = adjacency_mtx[k * size + col];
	if (tmp_b == INF)
		return;
	int cur = best + tmp_b;
	if (cur < adjacency_mtx[idx]) {
		adjacency_mtx[idx] = cur;
		paths[idx] = k;
	}
}

#pragma clang diagnostic pop