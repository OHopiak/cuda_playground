#ifndef CUDAFLOYD_PATHS_H
#define CUDAFLOYD_PATHS_H
#include <vector>
#include <iostream>

#define INF (1<<22)

typedef std::pair<std::pair<int, int>, int> path_item;

void show_path(int start, int end, const std::vector<path_item>& path, const int* graph, int size);

bool get_path(int curEdge, int nxtEdge, std::vector<path_item>& path, const int* graph, const int* paths, int size);

void get_full_paths(const int* adjacency_mtx, const int* paths, int size);

#endif //CUDAFLOYD_PATHS_H
