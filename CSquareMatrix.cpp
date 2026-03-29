#include <iostream>
#include <array>
#include <random>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <chrono>
#include <omp.h>


template <typename T, size_t Size>
class CSquareMatrix {
    private:
    std::array<std::array<T, Size>, Size> data_;
    
    public:
    CSquareMatrix(): data_{} {}


    CSquareMatrix(const CSquareMatrix& other) = default;


    CSquareMatrix& operator=(const CSquareMatrix& other) = default;


    ~CSquareMatrix() = default;



    std::array<T, Size>& operator[](size_t index) {
        return data_[index];
    }


    const std::array<T, Size>& operator[](size_t index) const {
        return data_[index];
    }


    void generateFullMatrix() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(1, 10);

        for (auto& row : data_) {
            for (auto& num : row) {
                num = dis(gen);
            }
        }
    }
};


template<typename T, size_t Size1, size_t Size2>
CSquareMatrix<T, Size1> multiplyMatrices(const CSquareMatrix<T, Size1>& mat1, const CSquareMatrix<T, Size2>& mat2) {
    if (Size1 != Size2) {
        throw std::invalid_argument("Matrices must have the same size for multiplication");
    }
    
    CSquareMatrix<T, Size1> result;

    #pragma omp parallel for
    for (int i = 0; i < Size1; i++) {
        for (int j = 0; j < Size1; j++) {
            for (int k = 0; k < Size1; k++) {
                result[i][j] += mat1[i][k]*mat2[i][j];
            }
        }
    }
    return result;
}


template<typename T, size_t Size1, size_t Size2>
void writeOriginalMatricesFile(const CSquareMatrix<T, Size1>& mat1, const CSquareMatrix<T, Size2>& mat2) {
    if (Size1 != Size2) {
        throw std::invalid_argument("Matrices must have the same size for multiplication");
    }

    std::ofstream file("original_matrices.txt");
    if (!file.is_open()) {
        throw std::runtime_error("Couldn't open the file");
    }

    for (size_t i = 0; i < Size1; i++) {
        for (size_t j = 0; j < Size1; j++) {
            file << mat1[i][j] << " ";
        }
        file << "\n";
    }

    file << '\n';

    for (size_t i = 0; i < Size2; i++) {
        for (size_t j = 0; j < Size2; j++) {
            file << mat2[i][j] << " ";
        }
        file << "\n";
    }

    file.close();
}


template<typename T, size_t Size1, size_t Size2>
void multiplitionCheck(const CSquareMatrix<T, Size1>& mat1, const CSquareMatrix<T, Size2>& mat2) {
    std::ofstream file("result_matrix.txt");
    if (!file.is_open()) {
        throw std::runtime_error("Couldn't open the file");
    }

    CSquareMatrix<int, Size1> res_mat;

    auto start_multiplication = std::chrono::high_resolution_clock::now();
    res_mat = multiplyMatrices(mat1, mat2);
    auto end_multiplication = std::chrono::high_resolution_clock::now();
    auto time_multiplication = std::chrono::duration_cast<std::chrono::microseconds>(end_multiplication - start_multiplication);
    file << "Multiplication time: " << time_multiplication.count() << " microseconds\n";
    file << "Number of operations: " << (2*Size1 - 1)*Size1*Size1 << "\n";

    for (size_t i = 0; i < Size1; i++) {
        for (size_t j = 0; j < Size1; j++) {
            file << res_mat[i][j] << " ";
        }
        file << "\n";
    }

    file.close();
}


int main() {
    omp_set_num_threads(12);

    CSquareMatrix<int, 1800> mat1;
    mat1.generateFullMatrix();
    CSquareMatrix<int, 1800> mat2;
    mat2.generateFullMatrix();
    try {
        writeOriginalMatricesFile(mat1, mat2);
        multiplitionCheck(mat1, mat2);
        system("python verification_of_the_result.py");
        std::cout << "Matrices are multiplied";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what();
    }
}
