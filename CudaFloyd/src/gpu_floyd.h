#ifndef CUDAFLOYD_GPU_FLOYD_H
#define CUDAFLOYD_GPU_FLOYD_H
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <device_launch_parameters.h>

#define _DTH cudaMemcpyDeviceToHost
#define _HTD cudaMemcpyHostToDevice
#define INF (1<<22)

#define BLOCK_SIZE 256


//CUDA GPU kernel/functions forward declaration
__global__ void wake_gpu(int reps);

void gpu_floyd(int* graph, int* paths, uint size);

__global__ void gpu_floyd_kernel(int k, int* adjacency_mtx, int* paths, int size);

#endif //CUDAFLOYD_GPU_FLOYD_H
