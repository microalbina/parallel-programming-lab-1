import numpy as np


def verifyMatrices():

    with open("D:/Projects/parallel-programming-lab-1/build/Debug/original_matrices.txt", "r", encoding = "utf-8") as file1:
        matrices = file1.read().split("\n\n")

        mat1 = np.array([[int(x) for x in row.split()] for row in matrices[0].strip().split("\n")])
        mat2 = np.array([[int(x) for x in row.split()] for row in matrices[1].strip().split("\n")])


    with open("D:/Projects/parallel-programming-lab-1/build/Debug/result_matrix.txt", "r", encoding = "utf-8") as file2:
        matrix_lines = file2.readlines()[2:]
        matrix = ''.join(matrix_lines).strip()

        cpp_mat = np.array([[int(x) for x in row.split()] for row in matrix.split("\n")])


    python_mat = np.matmul(mat1, mat2)
    if np.array_equal(python_mat, cpp_mat):
        print("Matrices are equal")
    else:
        print("Matrices are NOT equal")


if __name__ == "__main__":
    verifyMatrices()