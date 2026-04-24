#!/bin/bash
#SBATCH --job-name=run8
#SBATCH --ntasks-per-node=8
#SBATCH --time=00:30:00

mpirun ./matMulMPI