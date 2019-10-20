#include <algorithm>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <map>
#include <ctime>
#include <cassert>

#include "utils/timer.h"
#include "cpu_floyd.h"
#include "gpu_floyd.h"
#include "utils/files.h"
#include "paths.h"

#pragma clang diagnostic ignored "-Wuninitialized"

#define RANGE 997
#define RANDOM_GSIZE 700
#define FILE_GSIZE 8298 //the number of edges in data.txt if the file test is run
#define DO_TEST_RANDOM 1
#define DO_TEST_FROM_FILE 0

using namespace std;

void generate_random_graph(int* adjacency_mtx, int size, int range, int density);

int main()
{
	utils::Timer timer;
	srand(time(NULL));

	size_t num_bytes = RANDOM_GSIZE * RANDOM_GSIZE * sizeof(int);
	int* original_graph = new int[num_bytes];
	int* cpu_graph = new int[num_bytes];
	int* cpu_graph_path = new int[num_bytes];
	int* gpu_graph = new int[num_bytes];
	int* gpu_graph_path = new int[num_bytes];

	if (DO_TEST_RANDOM) {
		//init graph with values
		generate_random_graph(original_graph, RANDOM_GSIZE, RANGE, 25);

		cout << "Successfully created random highly connected graph in adjacency Matrix form with "
			 << RANDOM_GSIZE * RANDOM_GSIZE << " elements." << endl;
		cout << "Also created 2 pairs of distinct result Matrices to store "
			 << "the respective results of the CPU results and the GPU results." << endl;
	}

	for (int i = 0; i < RANDOM_GSIZE * RANDOM_GSIZE; i++) {
		cpu_graph[i] = gpu_graph[i] = original_graph[i];
		cpu_graph_path[i] = gpu_graph_path[i] = -1;
	}

	float cpu_time = 0, gpu_time = 0;

	cout << endl << "Floyd-Warshall on CPU underway:" << endl;
	timer.reset();
	cpu_floyd(cpu_graph, cpu_graph_path, RANDOM_GSIZE);
	cpu_time = timer.elapsed();
	cout << "CPU Timing: " << cpu_time << "ms" << endl;

	/*
	//wake up GPU from idle
	cout << endl << "Floyd-Warshall on GPU underway:" << endl;
	timer.reset();
	gpu_floyd(gpu_graph, gpu_graph_path, RANDOM_GSIZE);
	gpu_time = timer.elapsed();

	cout << "GPU Timing(including all device-host, host-device copies,"
		 << " device allocations and freeing of device memory): " << gpu_time << "ms"
		 << endl << endl;
	cout << "Verifying results of final adjacency Matrix and Path Matrix.\n";
	*/

	int same_adj_matrix = memcmp(cpu_graph, gpu_graph, num_bytes);
	if (same_adj_matrix == 0)
		cout << "Adjacency Matrices Equal!\n";
	else
		cout << "Adjacency Matrices Not Equal!\n";

	int same_path_matrix = memcmp(cpu_graph_path, gpu_graph_path, num_bytes);
	if (same_path_matrix == 0)
		cout << "Path reconstruction Matrices Equal!\n";
	else
		cout << "Path reconstruction Matrices Not Equal!\n";

	get_full_paths(gpu_graph, gpu_graph_path, RANDOM_GSIZE);
	//find out exact step-by-step shortest paths between vertices(if such a path exists)

	bool status = same_adj_matrix == 0 && same_path_matrix == 0;

	generate_result_file(status, cpu_time, gpu_time, RANDOM_GSIZE);

	delete original_graph;
	delete cpu_graph;
	delete cpu_graph_path;
	delete gpu_graph;
	delete gpu_graph_path;

	return 0;
}

void generate_random_graph(int* adjacency_mtx, int size, int range, int density)
{
	//density will be between 0 and 100, indication the % of number of directed edges in graph
	//range will be the range of edge weighting of directed edges
	int path_range = (100 / density);
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i == j) {
				//set adjacency_mtx[i][i]=0
				adjacency_mtx[i * size + j] = 0;
				continue;
			}
			int pr = rand() % path_range;
			//set edge random edge weight to random value, or to INF
			adjacency_mtx[i * size + j] = pr == 0 ? ((rand() % range) + 1) : INF;
		}
	}
}