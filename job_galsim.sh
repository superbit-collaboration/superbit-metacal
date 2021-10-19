#!/bin/sh
#SBATCH -t 12:00:00
#SBATCH --mem=8G
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -J test_pipe
#SBATCH -v 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmac.ftw@gmail.com
#SBATCH -o test_pipe.out
module load mpi
module load gcc/10.2

export OMP_NUM_THREADS=4


echo $PYTHONPATH
python ./superbit_lensing/process_sims.py
