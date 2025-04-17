#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <iomanip>
#include <omp.h>
#include <vector>


// Зчитує параметри
void parseArguments(int argc, char** argv, int& rows, int& cols, bool& show, int& numThreads) {
    if (argc >= 4) {
        rows = std::atoi(argv[1]);
        cols = std::atoi(argv[2]);
        char s = std::tolower(argv[3][0]);
        show = (s == 'y');
        if (argc >= 5) {
            numThreads = std::atoi(argv[4]);
        }
        std::cout << "rows=" << rows
                  << ", cols=" << cols
                  << ", show=" << (show ? "Y" : "N")
                  << ", threads=" << numThreads << "\n\n";
    } else {
        std::cout << "Default values:\n";
        std::cout << "rows=" << rows << ", cols=" << cols
                  << ", show=" << (show ? "Y" : "N")
                  << ", threads=" << numThreads << "\n\n";
    }
}

// Заповнення матриці
void fillMatrix(double* arr, int r, int c) {
    for (int i = 0; i < r * c; i++) {
        arr[i] = std::rand() % 10;
    }
}

// Вивід матриці
void printMatrix(const double* arr, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            std::cout << arr[i * c + j] << " ";
        }
        std::cout << "\n";
    }
}

// Послідовне обчислення
double computeFrobeniusNormSequential(const double* matrix, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; i++) {
        sum += matrix[i] * matrix[i];
    }
    return std::sqrt(sum);
}

// OpenMP з reduction
double computeFrobeniusNormOMP(const double* matrix, int size) {
    double sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < size; i++) {
        sum += matrix[i] * matrix[i];
    }
    return std::sqrt(sum);
}

// OpenMP ручне розбиття
double computeFrobeniusNormManualOptimized(const double* matrix, int size) {
    int numThreads = omp_get_max_threads();
    std::vector<double> localSums(numThreads, 0.0);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int chunkSize = size / numThreads;
        int start = tid * chunkSize;
        int end = (tid == numThreads - 1) ? size : start + chunkSize;

        double localSum = 0.0;
        for (int i = start; i < end; i++) {
            localSum += matrix[i] * matrix[i];
        }

        localSums[tid] = localSum; // без атомарок
    }

    double totalSum = 0.0;
    for (double val : localSums)
        totalSum += val;

    return std::sqrt(totalSum);
}


int main(int argc, char** argv) {
    int rowCount = 10, colCount = 10;
    bool showMatrixData = false;
    int numThreads = omp_get_max_threads();

    parseArguments(argc, argv, rowCount, colCount, showMatrixData, numThreads);
    omp_set_num_threads(numThreads);

    int size = rowCount * colCount;
    std::srand((unsigned)time(nullptr));
    double* matrix = new double[size];
    fillMatrix(matrix, rowCount, colCount);

    std::cout << std::fixed << std::setprecision(6);

    double startSeq = omp_get_wtime();
    double normSeq = computeFrobeniusNormSequential(matrix, size);
    double endSeq = omp_get_wtime();
    double timeSeq = endSeq - startSeq;
    std::cout << "Sequential Norm:  " << normSeq << " | Time: " << timeSeq << " s\n";

    double startOMP = omp_get_wtime();
    double normOMP = computeFrobeniusNormOMP(matrix, size);
    double endOMP = omp_get_wtime();
    double timeOMP = endOMP - startOMP;
    std::cout << "OpenMP Reduction: " << normOMP << " | Time: " << timeOMP << " s\n";
    std::cout << "Speedup (reduction): x" << (timeSeq / timeOMP) << "\n";

    double startManual = omp_get_wtime();
    double normManual = computeFrobeniusNormManualOptimized(matrix, size);
    double endManual = omp_get_wtime();
    double timeManual = endManual - startManual;
    std::cout << "Manual OpenMP:    " << normManual << " | Time: " << timeManual << " s\n";
    std::cout << "Speedup (manual):   x" << (timeSeq / timeManual) << "\n";

    if (showMatrixData) {
        std::cout << "\nMatrix Data:\n";
        printMatrix(matrix, rowCount, colCount);
    }

    delete[] matrix;
    return 0;
}
