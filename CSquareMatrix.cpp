#include <iostream>
#include <array>
#include <random>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <mpi.h>


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


template <typename T, size_t Size1, size_t Size2>
CSquareMatrix<T, Size1> multiplyMatricesMPI(const CSquareMatrix<T, Size1>& mat1, const CSquareMatrix<T, Size2>& mat2, int rank, int size) {
    if (Size1 != Size2) {
       throw std::invalid_argument("Matrices must have the same size for multiplication");
    }
    
    CSquareMatrix<T, Size1> result;

    int rows = Size1 / size;
    int remains = Size1 % size;

    int start_row = rank * rows;
    if (rank < remains) {
        start_row += rank;
        rows++;
    } else {
        start_row += remains;
    }
    int end_row = start_row + rows;

    CSquareMatrix<T, Size1> local_result;
    for (size_t i = start_row; i < end_row; i++) {
        for (size_t j = 0; j < Size1; j++) {
            T sum = 0;
            for (size_t k = 0; k < Size1; k++) {
                sum += mat1[i][k]*mat2[k][j];
            }
            local_result[i][j] = sum;
        }
    }

    if (rank == 0) {
        for (size_t i = start_row; i < end_row; i++) {
            for (size_t j = 0; j < Size1; j++) {
                result[i][j] = local_result[i][j];
            }
        }

        for (int process = 1; process < size; process++) {
            int new_rows = Size1 / size;
            int process_rows_start = process * new_rows;
            int process_rows = 0;

            if (process_rows_start < remains) {
                process_rows_start += process;
                process_rows = new_rows + 1;
            } else {
                process_rows_start += remains;
                process_rows = new_rows;
            }

            MPI_Recv(&result[process_rows_start][0], process_rows * Size1, MPI_INT, process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    } else {
        MPI_Send(&local_result[start_row][0], rows * Size1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    return result;
}


template <typename T, size_t Size>
void generateFullMatrixMPI(CSquareMatrix<T, Size>& mat, int rank, int size) {
    if (rank == 0) {
        mat.generateFullMatrix();

        for (size_t i = 1; i < size; i++) {
            MPI_Send(&mat[0][0], Size * Size, MPI_INT, i, 0, MPI_COMM_WORLD);
        }

    } else {
        MPI_Recv(&mat[0][0], Size * Size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
}


template <typename T, size_t Size1, size_t Size2>
void writeOriginalMatricesFile(const CSquareMatrix<T, Size1>& mat1, const CSquareMatrix<T, Size2>& mat2, int rank) {
    if (Size1 != Size2) {
            throw std::invalid_argument("Matrices must have the same size for multiplication");
    }

    if (rank == 0) {
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
}


template <typename T, size_t Size1, size_t Size2>
void multiplitionCheckMPI(const CSquareMatrix<T, Size1>& mat1, const CSquareMatrix<T, Size2>& mat2, int rank, int size) {
    if (Size1 != Size2) {
            throw std::invalid_argument("Matrices must have the same size for multiplication");
    }

    CSquareMatrix<int, Size1> res_mat;
    MPI_Barrier(MPI_COMM_WORLD);
    auto start_multiplication = MPI_Wtime();
    res_mat = multiplyMatricesMPI(mat1, mat2, rank, size);
    MPI_Barrier(MPI_COMM_WORLD);
    auto end_multiplication = MPI_Wtime();

    if (rank == 0) {
        std::ofstream file("result_matrix.txt");
        if (!file.is_open()) {
            throw std::runtime_error("Couldn't open the file");
        }

        auto time_multiplication = (end_multiplication - start_multiplication) * 1000000;
        file << "Multiplication time: " << time_multiplication << " microseconds\n";
        file << "Number of operations: " << (2*Size1 - 1)*Size1*Size1 << "\n";

        for (size_t i = 0; i < Size1; i++) {
            for (size_t j = 0; j < Size1; j++) {
                file << res_mat[i][j] << " ";
            }
            file << "\n";
        }

        file.close();
    }
}


int main(int argc, char* argv[]) {

    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    CSquareMatrix<int, 1600> mat1;
    CSquareMatrix<int, 1600> mat2;
    generateFullMatrixMPI(mat1, rank, size);
    generateFullMatrixMPI(mat2, rank, size);

    try {
        writeOriginalMatricesFile(mat1, mat2, rank);
        multiplitionCheckMPI(mat1, mat2, rank, size);
        system("python verification_of_the_result.py");
        if (rank == 0) {
        std::cout << "Matrices are multiplied";
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what();
    }

    MPI_Finalize();
}
