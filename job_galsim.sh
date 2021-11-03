#!/bin/sh
#SBATCH -t 2:00:00
#SBATCH -N 1 
#SBATCH -n 8
#SBATCH --mem-per-cpu=5g
#SBATCH -J test-sims
#SBATCH -v 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmac.ftw@gmail.com
#SBATCH -o slurm_outfiles/forecast-sims-omp8.out


module load mpi
module load gcc/10.2


echo $PYTHONPATH
echo $OMP_NUM_THREADS

export OMP_NUM_THREADS=8

python /users/jmcclear/data/superbit/superbit-metacal/superbit_lensing/process_sims.py

