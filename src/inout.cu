#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#include "inout.cuh"

void loadMatrix(const std::string& filename, float* data, int rows, int cols) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法打开文件: " << filename << std::endl;
        exit(1);
    }
    for (int i = 0; i < rows * cols; i++) {
        file >> data[i];
    }
    file.close();
}

void saveMatrix(const std::string& filename, float* data, int rows, int cols) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "无法保存文件: " << filename << std::endl;
        exit(1);
    }
    file.precision(6);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << data[i * cols + j] << (j < cols - 1 ? " " : "");
        }
        file << "\n";
    }
    file.close();
}