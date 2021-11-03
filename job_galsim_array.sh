#!/bin/sh
#SBATCH -t 6:00:00
#SBATCH --mem=50G
#SBATCH --nodes 2
#SBATCH -J cluster5-v3
#SBATCH -v 
#SBATCH --array=1-9
# Use '%A' for array-job ID, '%J' for job ID and '%a' for task ID
#SBATCH -o array-job-outputs/cl5-v3-arrayjob-%a.out
#SBATCH -e array-job-outputs/cl5-v3-arrayjob-%a.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmac.ftw@gmail.com


### Now do serial job
#mkdir -p array-job-outputs2/ --> needs to be exist for outputs to be saved
echo "Starting job $SLURM_ARRAY_TASK_ID on $HOSTNAME"

GALSIM_DIR=/users/jmcclear/data/superbit/superbit-metacal/GalSim
CONFIG_FILE=$GALSIM_DIR/config_files/superbit_parameters_forecast.yaml
OUT_DIR=/users/jmcclear/data/superbit/mock-data-forecasting/cluster5/v1/round$SLURM_ARRAY_TASK_ID

### Generate a random seed for noise
seedfunc(){
    shuf -i 11111111-99999999 -n 1
    }
NOISE_SEED="$(seedfunc)"
STARS_SEED="$(seedfunc)"
GALOBJ_SEED="$(seedfunc)"
CLUSTER_SEED="$(seedfunc)"

python mock_superBIT_data.py config_file=$CONFIG_FILE outdir=$OUT_DIR noise_seed=$NOISE_SEED galobj_seed=$GALOBJ_SEED cluster_seed=$CLUSTER_SEED stars_seed=$STARS_SEED
