#include <iostream>
#include <array>
#include <random>
#include <ctime>


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


    void print() {
        for (const auto& row : data_) {
            for (const auto& num : row) {
                std::cout << num << " ";
            }
            std::cout << "\n";
        }
    }


    void generateFullMatrix() {
        static std::mt19937 gen{std::random_device{}()};
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
    for (size_t i = 0; i < Size1; i++) {
        for (size_t j = 0; j < Size1; j++) {
            for (size_t k = 0; k < Size1; k++) {
                result[i][j] += mat1[i][k]*mat2[k][j];
            }
        }
    }
    return result;
}

int main() {
    CSquareMatrix<int, 10> matrix1;
    matrix1.generateFullMatrix();
    matrix1.print();
    std::cout << "\n";
    CSquareMatrix<int, 10> matrix2;
    matrix2.generateFullMatrix();
    matrix2.print();
    std::cout << "\n";
    CSquareMatrix<int, 10> matrix3;
    matrix3 = multiplyMatrices(matrix1, matrix2);
    matrix3.print();
}
