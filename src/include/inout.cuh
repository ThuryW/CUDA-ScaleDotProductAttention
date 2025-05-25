// inout.cuh
#ifndef INOUT_CUH
#define INOUT_CUH

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

void loadMatrix(const std::string& filename, float* data, int rows, int cols);

void saveMatrix(const std::string& filename, float* data, int rows, int cols);

#endif