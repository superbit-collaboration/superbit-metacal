#!/bin/sh
#SBATCH -t 23:59:59
#SBATCH -N 1
#SBATCH -n 18
#SBATCH --mem-per-cpu=10g
#SBATCH --partition=short
#SBATCH -J Abell3571_b
#SBATCH -v
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jmac.ftw@gmail.com
#SBATCH -o Abell3571_b.out
#SBATCH -e Abell3571_b.err


source /work/mccleary_group/miniconda3/etc/profile.d/conda.sh

conda activate sbmcal_139

###
### Define some environmental variables
###

export OUTDIR='/scratch/j.mccleary/real-data-2023/Abell3571/b/out'
export CATDIR='/work/mccleary_group/superbit/real-data-2023/Abell3571/det/cat/'
export DATADIR='/work/mccleary_group/superbit/real-data-2023/'
export CODEDIR='/work/mccleary_group/superbit/superbit-metacal/superbit_lensing'


###
### Path checking
###
export PATH=$PATH:'/work/mccleary_group/Software/texlive-bin/x86_64-linux'
echo $PATH
echo $PYTHONPATH
dirname="slurm_outfiles"
if [ ! -d "$dirname" ]
then
     echo " Directory $dirname does not exist. Creating now"
     mkdir -p -- "$dirname"
     echo " $dirname created"
 else
     echo " Directory $dirname exists"
 fi

 echo "Proceeding with code..."


###
### Commands!
###


### Medsmaker

#python $CODEDIR/medsmaker/scripts/process_2023.py Abell3571 b $DATADIR -outdir=$OUTDIR \
#-psf_mode=piff -psf_seed=33876300 --meds_coadd -star_config_dir $CODEDIR/medsmaker/configs \
#--select_truth_stars

### Metacal

python $CODEDIR/metacalibration/ngmix_fit_superbit3.py $OUTDIR/Abell3571_b_meds.fits \
$OUTDIR/Abell3571_b_mcal.fits -outdir=$OUTDIR -n 48 -seed=84700353 --overwrite

### Annular & shear

python $CODEDIR/shear_profiles/make_annular_catalog.py $DATADIR Abell3571 $OUTDIR/Abell3571_b_mcal.fits \
$OUTDIR/Abell3571_b_annular.fits -outdir=$OUTDIR --overwrite -cluster_redshift=0.0393 \
-redshift_cat=$DATADIR/Abell3571/Abell3571_detection_cat_redshits.fits \
-rmin=300 -rmax=4375 -nbins=15


