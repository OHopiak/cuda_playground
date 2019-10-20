#include "files.h"

//reads in edge list from file
int read_from_file(int* adjacency_mtx, int size)
{
	int num_edges = 0;
	std::ifstream readfile("data.txt");
	if(!readfile.good()) {
		return 0;
	}
	std::string line;
	int v0 = 0, v1 = 0;
	while (getline(readfile, line)) {
		std::istringstream buffer(line);
		buffer >> v0 >> v1;
		adjacency_mtx[v0 * size + v1] = 1;
		num_edges++;
	}
	readfile.close();
	return num_edges;
}

void generate_result_file(bool success, float cpu_time, float gpu_time, int size)
{
	std::ofstream file("result.txt");
	if (!file.is_open())
		return;

	if (!success) {
		file << "Error in calculation!\n";
	} else {
		file << "Success! The GPU Floyd-Warshall result and the CPU Floyd-Warshall"
			 << " results are identical(both final adjacency matrix and path matrix).\n\n";
		file << "size= " << size << " , and the total number of elements(for Adjacency Matrix and Path Matrix) was "
			 << size * size << " .\n";
		file << "Matrices are int full dense format(row major) with a minimum of " << (size * size) / 4
			 << " valid directed edges.\n\n";
		file << "The CPU timing for all was " << cpu_time / 1000.0f
			 << " seconds, and the GPU timing(including all device memory operations(allocations,copies etc) ) for all was "
			 << gpu_time / 1000.0f << " seconds.\n";
		file << "The GPU result was " << cpu_time / gpu_time << " faster than the CPU version.\n";
	}
	file.close();
}
