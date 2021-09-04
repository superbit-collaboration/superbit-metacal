import os
import glob
import sys
import pdb

def main(argv):
    sextractor_config_path = '/Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/'
    #all_fits_files=glob.glob("/Users/jemcclea/Research/GalSim/examples/output-bandpass/mockSuperbit_scaled_empiricalPSF_???_?.fits")
    all_fits_files=glob.glob("/Users/jemcclea/Research/SuperBIT/A2218/Clean/other_filters/dwb*.fits")
    
    for fits in all_fits_files:
        detection_file=fits
        weight_file ='/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/mask_files/supermask.fits' #for want of st better...
        #weight_arg = '-WEIGHT_IMAGE '+weight_file
        weight_arg = '-WEIGHT_TYPE NONE'
        config_arg = sextractor_config_path+'sextractor.config'
        param_arg = '-PARAMETERS_NAME '+sextractor_config_path+'sextractor.param'
        nnw_arg = '-STARNNW_NAME '+sextractor_config_path+'default.nnw'
        filter_arg = '-FILTER_NAME '+sextractor_config_path+'default.conv'

        try:

            outname=fits.split('dwb_')[1].replace('.fits','.ldac')
            bkgname=fits.split('dwb_')[1].replace('.fits','aper.fits')

        except:
            #outname=fits.split('mockSuperbit_')[1].replace('.fits','.ldac')
            #bkgname=fits.split('mockSuperbit_')[1].replace('.fits','.sub.fits')
            outname=fits.replace('.fits','.ldac')
            bkgname=fits.replace('.fits','.aper.fits')
            #outname='A2218_coadd_cat2.fits'
        name_arg ='-CATALOG_NAME ' + outname
        bkg_arg = '-CHECKIMAGE_NAME ' + bkgname
        
        cmd = ' '.join(['sex',detection_file,name_arg,bkg_arg, weight_arg, param_arg,nnw_arg,filter_arg,'-c',config_arg])
        print("sex cmd is " + cmd)
        os.system(cmd)
        
if __name__ == "__main__":
    
    main(sys.argv)


"""
work_path='/Users/jemcclea/Research/GalSim/examples/output-bandpass'
outfile_name='bandpass_scaled_empiricalPSF_coadd.fits'; weightout_name='bandpass_scaled_empiricalPSF_coadd.weight.fits'
image_files=glob.glob('/Users/jemcclea/Research/GalSim/examples/output-bandpass/mockSuperbit_scaled_empiricalPSF_???_?.fits')
image_args = ' '.join(image_files)
detection_file = os.path.join(work_path,outfile_name) # This is coadd
weight_file = os.path.join(work_path,weightout_name) # This is coadd weight
config_arg = '-c /Users/jemcclea/Research/SuperBIT_2019/superbit-ngmix/superbit/astro_config/swarp.config'

weight_arg = '-WEIGHT_IMAGE /Users/jemcclea/Research/SuperBIT_2019/superbit-ngmix/scripts/output-real2/supermask.fits' 
outfile_arg = '-IMAGEOUT_NAME '+ detection_file + ' -WEIGHTOUT_NAME ' + weight_file
cmd = ' '.join(['swarp ',image_args,weight_arg,outfile_arg,config_arg])
print("swarp cmd is " + cmd)

os.system(cmd)


swarp round?/superbit_gaussJitter_0??.fits -IMAGEOUT_NAME forecast_cl5.fits -c ~/Research/SuperBIT/superbit-ngmix/superbit/astro_config/swarp.config

sex forecast_cl3.fits -WEIGHT_IMAGE coadd.weight.fits -CATALOG_NAME cl3forecast_coadd_cat.ldac -PARAMETERS_NAME /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/sextractor.param -STARNNW_NAME /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/default.nnw -FILTER_NAME /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/default.conv -c /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/sextractor.mock.config -CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME apertures.fits

sex superbit_gaussJitter_001.fits -WEIGHT_TYPE NONE -CATALOG_NAME double_srcs_coadd_cat.ldac -PARAMETERS_NAME /Users/jemcclea/Resrch/SuperBIT/superbit-ngmix/superbit/astro_config/sextractor.param -STARNNW_NAME /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/default.nnw -FILTER_NAME /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/default.conv -c /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/sextractor.empirical.config -CHECKIMAGE_TYPE APERTURES -CHECKIMAGE_NAME apertures.fits



"""
