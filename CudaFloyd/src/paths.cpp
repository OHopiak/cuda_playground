#include "paths.h"

void show_path(int start, int end, const std::vector<path_item>& path, const int* graph, int size)
{
	std::cout << std::endl << "Here is the shortest cost path from " << start << " to " << end
			  << ", at a total cost of "
			  << graph[start * size + end] << ".\n";
	for (size_t i = path.size() - 1; i >= 0; --i) {
		std::cout << "From " << path[i].first.first << " to " << path[i].first.second
				  << " at a cost of " << path[i].second << std::endl;
	}
	std::cout << '\n';
}

bool get_path(int curEdge, int nxtEdge, std::vector<path_item>& path, const int* graph, const int* paths, int size)
{
	int curIdx = curEdge * size + nxtEdge;
	if (graph[curIdx] >= INF) return false;
	if (paths[curIdx] == -1) {
		//end of backwards retracement
		path.emplace_back(std::make_pair(curEdge, nxtEdge), graph[curIdx]);
		return true;
	} else {
		//record last edge cost and move backwards
		path.emplace_back(std::make_pair(paths[curIdx], nxtEdge), graph[paths[curIdx] * size + nxtEdge]);
		return get_path(curEdge, paths[curIdx], path, graph, paths, size);
	}
}

void get_full_paths(const int* adjacency_mtx, const int* paths, int size)
{
	int start_vertex = -1, end_vertex = -1;
	std::vector<path_item> path{};
	do {
		path.clear();
		std::cout << "Enter start and end vertices (enter negative number to exit): ";
		std::cin >> start_vertex >> end_vertex;
		if (start_vertex < 0 || start_vertex >= size || end_vertex < 0 || end_vertex >= size) break;

		bool path_found = get_path(start_vertex, end_vertex, path, adjacency_mtx, paths, size);
		if (path_found) {
			show_path(start_vertex, end_vertex, path, adjacency_mtx, size);
		} else {
			std::cout << std::endl << "There does not exist valid a path between " << start_vertex << ", and "
					  << end_vertex << std::endl;

		}
	} while (true);
}
