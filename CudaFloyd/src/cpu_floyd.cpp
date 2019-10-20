#include "cpu_floyd.h"

//standard O(n^3) algorithm
void cpu_floyd(int* graph, int* paths, int size)
{
	for (int k = 0; k < size; ++k) {
		for (int i = 0; i < size; ++i) {
			for (int j = 0; j < size; ++j) {
				int current_location = i * size + j;
				int local_a = i * size + k;
				int local_b = k * size + j;
				if (graph[current_location] > (graph[local_a] + graph[local_b])) {
					graph[current_location] = (graph[local_a] + graph[local_b]);
					paths[current_location] = k;
				}
			}
		}
	}
}
