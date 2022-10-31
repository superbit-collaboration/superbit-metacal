#!/bin/bash
#PBS -N 12-cl_m2.5e14_z0.075
#PBS -q array-sn
#PBS -l select=1:ncpus=12:mem=144gb
#PBS -l walltime=16:00:00
#PBS -J 0-29
#PBS -o job_stdout.txt
#PBS -e job_stderr.txt
#PBS -v summary=true

### NOTE: Replace values above to match your run params.
###       Running w/ ncores < 1 node can help during
###       high cluster utilization
 
### Load modules into your environment
### NOTE: Replace w/ your own conda env location if desired
. /projects/superbit/miniconda3/etc/profile.d/conda.sh
conda activate sbmcal_139 # NOTE: Replace w/ your desired env

### Setup vars
### NOTE: Replace w/ your repo dir if not using the main one
REPO_DIR=/projects/superbit/repos/superbit-metacal
RUN_NAME={RUN_NAME}

# NOTE: Examples below, fill with your own values
MASS="m2.5e14"
Z="z0.075"
CLUSTER=cl_"$MASS"_$Z
REAL="r$PBS_ARRAY_INDEX"

RUN_DIR=$REPO_DIR/runs/$RUN_NAME/$CLUSTER/$REAL/

CONFIG_FILE="$RUN_NAME"_"$CLUSTER".yaml

### Run executable
cd $REPO_DIR/superbit_lensing/

python run_pipe.py $RUN_DIR/$CONFIG_FILE

### Cleanup & compress
rm $RUN_DIR/*.sub.fits
rm $RUN_DIR/*.sgm.fits
rm $RUN_DIR/*.ldac
fpack $RUN_DIR/*_0*.fits
rm $RUN_DIR/*_0*.fits
fpack $RUN_DIR/*meds*.fits
rm $RUN_DIR/*meds*.fits
fpack $RUN_DIR/weight_files/*weight.fits
rm $RUN_DIR/weight_files/*weight.fits
fpack $RUN_DIR/mask_files/*mask.fits
rm $RUN_DIR/mask_files/*mask.fits
