#ifndef CUDAFLOYD_FILES_H
#define CUDAFLOYD_FILES_H
#include <fstream>
#include <sstream>

//other optional utility functions
int read_from_file(int* adjacency_mtx, int size);

void generate_result_file(bool success, float cpu_time, float gpu_time, int size);

#endif //CUDAFLOYD_FILES_H
