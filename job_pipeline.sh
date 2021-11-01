#!/bin/sh
#SBATCH -t 6:00:00
#SBATCH -N 1 
#SBATCH -n 18
#SBATCH --mem-per-cpu=5g
#SBATCH -J psfex_cutout_mcal
#SBATCH -v 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmac.ftw@gmail.com
#SBATCH -o slurm_outfiles/psfex_poly3_medcombine_tpv.out


#module load mpi
#module load gcc/10.2


echo $PYTHONPATH
echo $OMP_NUM_THREADS


python /users/jmcclear/data/superbit/superbit-metacal/superbit_lensing/process_all.py
#python /users/jmcclear/data/superbit/superbit-metacal/superbit_lensing/process_all2.py                                                                              
#python /users/jmcclear/data/superbit/superbit-metacal/superbit_lensing/process_all3.py
