#include <mpi.h>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <iostream>
#include <array>
#include <algorithm>
#include <random>      
#include <type_traits>

template <typename T, size_t N>
class Matrix{
    std::array<T, N*N> arr;
    public:
    Matrix(): arr{}{}
    Matrix(const T val){
        arr.fill(val);
    }

    Matrix(const T& min, const T& max, const unsigned int seed){
        std::mt19937 gen(seed);
        std::uniform_int_distribution<T> dist(min, max);
        for(size_t i = 0; i < N*N; i++){
            arr[i] = dist(gen);
        }
    }

    size_t size()const{
        return N;
    }
    
    T* data(){
        return arr.data();
    }

    const T* data() const{
        return arr.data();
    }

    T& operator()(const size_t i, const size_t j){
        return arr[i*N + j];
    }

    const T& operator()(const size_t i, const size_t j) const{
        return arr[i*N + j];
    }

    Matrix operator+(const Matrix& m)const{
        Matrix<T, N> res;
        for (size_t i = 0; i < N; i++){
            for(size_t j = 0; j < N; j++)
                res(i, j) = (*this)(i, j) + m(i, j);
        }
        return res;
    }

    Matrix<T, N> operator*(const T c) const{
        Matrix<T, N> res;
        for (size_t i = 0; i < N; i++){
            for(size_t j = 0; j < N; j++)
                res(i, j) = (*this)(i, j)*c;
        }
        return res;
    }

    friend Matrix<T, N> operator*(const T c, const Matrix<T, N>& m){
        return m*c;
    }

    Matrix<T, N> operator*(const Matrix& m) const{
        Matrix<T, N> res;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    res(i, j) += (*this)(i, k) * m(k, j);
                }
            }
        }
        return res;
    }

    Matrix<T, N> mpi_multi(const Matrix<T, N>& m) const {
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        if (N % size != 0) {
            if (rank == 0)
                std::cerr << "Error\n";
            return Matrix<T, N>();
        }

        int rows = N / size;
        int start_row = rank * rows;

        Matrix<T, N> loc_res;
        
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++)
                loc_res(i, j) = 0;
        
        for (int i = start_row; i < start_row + rows; i++) {
            for (int j = 0; j < N; j++) {
                T sum = 0;
                for (int k = 0; k < N; k++) {
                    sum += (*this)(i, k) * m(k, j);
                }
                loc_res(i, j) = sum;
            }
        }

        Matrix<T, N> res;
        
        MPI_Datatype mpi_type = MPI_INT;

        MPI_Reduce(loc_res.data(), res.data(), N * N, mpi_type, MPI_SUM, 0, MPI_COMM_WORLD);

        return res;
    }
    
    friend std::ostream& operator<<(std::ostream& os, const Matrix<T, N>& m){
        for(size_t i = 0; i<N; i++){
            for(size_t j = 0; j < N; j++){
                os << m(i, j) << " ";
            }
            os << "\n";
        }
        return os;
    }

};

int main(int argc, char** argv){
    MPI_Init(&argc, &argv);
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const size_t N = 1600;
    Matrix<int, N> m1, m2;
    
    if (rank == 0) {
        m1 = Matrix<int, N>(1, 100, 8);
        m2 = Matrix<int, N>(-134, 670, 45);
        
        std::ofstream out_begin("begin.txt");
        if (out_begin.is_open()) {
            out_begin << "Matrix A:\n" << m1 << "Matrix B:\n" << m2;
            out_begin.close();
        }
    }
    
    MPI_Bcast(m1.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(m2.data(), N * N, MPI_INT, 0, MPI_COMM_WORLD);
    
    auto start = std::chrono::high_resolution_clock::now();
    auto res = m1.mpi_multi(m2);
    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    if (rank==0) {
	std::cout << N << "   /n   ";
	std::cout << time.count()/1000.0 << " ms/n";
    }
    if (rank == 0) {
        
        std::ofstream out("end.txt");
        if (out.is_open()) {
            out << "Result Matrix:\n" << res << "\n";
            out << "Matrix's size: " << N << "x" << N << "\n";
            out << "Time: " << time.count()/1000.0 << " ms" << std::flush; 
            out.close();
        }
    }
    
    MPI_Finalize();
    return 0;
}