#!/bin/sh
#SBATCH -t 00:30:00
#SBATCH --partition=debug
#SBATCH --mem-per-cpu=5GB
#SBATCH --nodes 1
#SBATCH -J smallshear
#SBATCH -v
#SBATCH --array=1-29
# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -o array-job-outputs/arrayjob-%a.out
#SBATCH -e array-job-outputs/arrayjob-%a.err

module load texlive/2018

CODE_DIR=/gpfs/data/idellant/jmcclear/superbit/superbit-metacal/SmallShearCode
SEX_CONFIG_DIR=/gpfs/data/idellant/jmcclear/superbit/superbit-metacal/superbit_lensing/medsmaker/superbit/astro_config/
NFW_DIR=/gpfs/data/idellant/jmcclear/superbit/mock-data-forecasting/nfw_truth/

SHEAR_CUTOFF=0.08
CLUSTER_NAME=cl_m1.6e15_z0.45

RUN_NAME=cosmos3_ss2

DATA_DIR=/users/jmcclear/data/superbit/mock-data-forecasting/cosmos3_ss/$CLUSTER_NAME

RUN_DIR=$DATA_DIR/r$SLURM_ARRAY_TASK_ID
RUN_DIR_ZERO=$DATA_DIR/r0


##
## Mock coadd catalogs got deleted
##


sex $RUN_DIR/"$RUN_NAME"_mock_coadd.fits -CATALOG_NAME $RUN_DIR/"$RUN_NAME"_mock_coadd_cat.ldac -CHECKIMAGE_NAME  $RUN_DIR/"$RUN_NAME"_mock_coadd.sub.fits,$RUN_DIR/"$RUN_NAME"_mock_coadd.sgm.fits -PARAMETERS_NAME $SEX_CONFIG_DIR/sextractor.param -STARNNW_NAME $SEX_CONFIG_DIR/default.nnw -FILTER_NAME $SEX_CONFIG_DIR/default.conv -c $SEX_CONFIG_DIR/sextractor.mock.config -WEIGHT_IMAGE $RUN_DIR/"$RUN_NAME"_mock_coadd.weight.fits -WEIGHT_TYPE MAP_WEIGHT

python $CODE_DIR/small_shear_runner.py $RUN_DIR/"$RUN_NAME"_mock_coadd_cat.ldac $RUN_DIR/"$RUN_NAME"_mcal.fits $RUN_DIR/"$RUN_NAME"_annular_redo.fits -outdir=$RUN_DIR -run_name="$RUN_NAME" -nfw_file=$NFW_DIR/nfw_"$CLUSTER_NAME".fits --overwrite --vb -truth_file=$RUN_DIR/"$RUN_NAME"_truth.fits -shear_cutoff="$SHEAR_CUTOFF"

##
## Do this too b/c code gets confused with r0
##

sex $RUN_DIR_ZERO/"$RUN_NAME"_mock_coadd.fits -CATALOG_NAME $RUN_DIR_ZERO/"$RUN_NAME"_mock_coadd_cat.ldac -CHECKIMAGE_NAME  $RUN_DIR_ZERO/"$RUN_NAME"_mo\
ck_coadd.sub.fits,$RUN_DIR_ZERO/"$RUN_NAME"_mock_coadd.sgm.fits -PARAMETERS_NAME $SEX_CONFIG_DIR/sextractor.param -STARNNW_NAME $SEX_CONFIG_DI\
R/default.nnw -FILTER_NAME $SEX_CONFIG_DIR/default.conv -c $SEX_CONFIG_DIR/sextractor.mock.config -WEIGHT_IMAGE $RUN_DIR_ZERO/"$RUN_NAME"_mock\
_coadd.weight.fits -WEIGHT_TYPE MAP_WEIGHT

python $CODE_DIR/small_shear_runner.py $RUN_DIR_ZERO/"$RUN_NAME"_mock_coadd_cat.ldac $RUN_DIR_ZERO/"$RUN_NAME"_mcal.fits $RUN_DIR_ZERO/"$RUN_NAME"_annul\
ar_redo.fits -outdir=$RUN_DIR_ZERO -run_name="$RUN_NAME" -nfw_file=$NFW_DIR/nfw_"$CLUSTER_NAME".fits --overwrite --vb -truth_file=$RUN_DIR_ZERO/"$RUN_NA\
ME"_truth.fits -shear_cutoff="$SHEAR_CUTOFF"
