import numpy as np
import sys
import os
import galsim
import galsim.des
import pdb
import scipy
from numpy import random
from astropy.table import Table
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
#plt.ion()
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colorbar import Colorbar



"""
 - Purpose of this code is to... find the place at which the breakdown of galaxy detection occurs in b filter

 - What am I trying to prove?  Which sky background matches expection that a mag=19 galaxy can  
   be detected on a single 300s exposure, while a mag=24 galaxy can be detected in 4000s of exposure

 - Some paramter notes. TO DO: build a class that can access this information
     - mean HLR of sample = 0.609"
     - mean HLR of mag 24 gal = 0.78"; median HLR of mag 24 gal = 0.375"
     - mean HLR of mag 19.5 gal = 1.08"; median HLR of mag 19.5 gal = 1.25"
     - mean b flux of mag 19.5 gal = 16.9 ADU/s; mean shape flux of mag 19.5 gal = 33.5 ADU/s
     - mean b flux of mag 24 gal = 0.268 ADU/s; mean shape flux of mag 24 gal = 0.422 ADU/s
     - AG base_sky_level in b = 0.039 ADU/s/pix // shape =  0.125 ADU/s/pix
     - MS base_sky_level in b = 0.12 ADU/s/pix // shape = 0.303 ADU/s/pix


"""

#cosmos=Table.read('data/cosmos2015_cam2021_filt2021.csv')

###
### Set some parameters
###

# define an inclinedSersic galaxy

base_gal_flux = 0.422                # flux of m=24 galaxy (ADU/s)
half_light_radius = 0.4              # np.mean(cosmos['hlr_cosmos10']*0.03)
n = 2.479                            # cosmos['n_sersic_cosmos10'][index]
scale_h_over_r = 0.53                # np.mean(cosmos['q_cosmos10']) for m=24 gal
inclination = (np.pi - 0.202)*galsim.radians  # np.mean(cosmos['phi_cosmos10']) = 0.202

# Characterizing IMX455

read_noise = 1.8                     # e-
gain = 0.81                          # e-/ADU, which tells GalSim what to do with RN in e-.
pixel_scale = 0.144                  # arcsec / pixel
psf_fwhm = 0.33                      # arcsec
base_sky_level = 0.303               # ADU / s / pix

# Characterizing observations

band = 'shape'                           # name of filter in which galaxy is being observed
n_obs = 30                               # number of observations of duration 'exp_time'
exp_time = 300                           # s
sky_level =  base_sky_level * exp_time   # ADU / pix
gal_flux = base_gal_flux * exp_time      # ADU
seed = random.randint(11111111,99999999)


###      
### Draw galaxy and PSF
###

gal = galsim.InclinedSersic(n = n,
                                flux=gal_flux,
                                half_light_radius=half_light_radius,
                                inclination=inclination,
                                scale_h_over_r=scale_h_over_r
                                )

psf = galsim.Gaussian(flux=1., fwhm=psf_fwhm)
final = galsim.Convolve([gal, psf])

gal_only_stamp = gal.drawImage(scale=pixel_scale)
psf_stamp = psf.drawImage(scale=pixel_scale) 
obs_stamp = final.drawImage(scale=pixel_scale,nx=50,ny=50)


## Create noisy observations

image_obslist = [] 
for obs in range(n_obs):
    ##
    ## Define noise, add it to stamps
    ##
    this_gal_stamp = final.drawImage(scale=pixel_scale,nx=50,ny=50)
    ud = galsim.UniformDeviate(seed+obs+1)
    noise = galsim.CCDNoise(rng=ud,sky_level=sky_level,gain=gain,read_noise=read_noise)
    this_gal_stamp.addNoise(noise)
    image_cutout = this_gal_stamp.array
    image_obslist.append(image_cutout)
    del(this_gal_stamp)


####################################################
## Uncomment if using script in interactive mode
"""
## Inspect single galaxy observation before noise
plt.figure()
plt.imshow(obs_stamp.array)
plt.title('PSF-convolved galaxy\n%d s observation (noise-free)' % int(exp_time))
plt.colorbar()
plt.savefig('diagnostics_plots/m24_noisefree_gal.png')

# Inspect single noisy observation
plt.figure()
plt.imshow(image_cutout)
plt.title('Single galaxy observation \nbase sky bkg = %.3f exptime = %d s' % (base_sky_level,int(exp_time)))
plt.colorbar()
plt.savefig('diagnostics_plots/m24_hisky_single300sObs.png')

# See what coadded image might look like
junk_avg = np.mean(image_obslist,axis=0)
plt.figure()
plt.imshow(junk_avg)
plt.colorbar()
plt.title('Noisy galaxy nexp=%d "coadd"\nbase sky bkg = %.3f exptime = %d s' % (n_obs,base_sky_level,int(exp_time)))
plt.savefig('diagnostics_plots/m24_hisky_80exposures_meanCombine.png')

"""


###########################################################
###########################################################

# Set up figure and image grid
rcParams['mpl_toolkits.legacy_colorbar']=False

fig = plt.figure(figsize=(15, 5))
grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(1,3),
                 axes_pad=0.5,
                 #share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="7%",
                 cbar_pad=0.15,
                 )
axes=[]
for ax in grid: 
     axes.append(ax) 
ax1=axes[0];ax2=axes[1];ax3=axes[2]

## Show theoretical galaxy model
ax1.imshow(obs_stamp.array)
ax1.set_title('PSF-convolved galaxy HLR=%.2f"\n%d s observation (noise-free)' \
                  % (half_light_radius,int(exp_time)),fontsize=10)

## Show single observation of galaxy
ax2.imshow(image_cutout,vmax=(image_cutout.max()*0.90))
ax2.set_title('Single galaxy observation \nbase sky bkg = %.3f exptime = %d s'\
                  % (base_sky_level,int(exp_time)),fontsize=10)

## Make & show a "coadd" image with simple average
junk_avg = np.mean(image_obslist,axis=0)
coadd = ax3.imshow(junk_avg)
ax3.set_title('Noisy galaxy nexp=%d "coadd"\nbase sky bkg = %.3f exptime = %d s'\
                  % (n_obs,base_sky_level,int(exp_time)),fontsize=10)
ax3.cax.colorbar(coadd)

outdir = '.'#/diagnostics_plots'
outroot = "{0}_mag24gal_hlr{4}_{1}s_{2}nObs_{3}SkyLevel_2.png"\
  .format(band,exp_time,n_obs,base_sky_level,half_light_radius)
outname = os.path.join(outdir,outroot)
fig.savefig(outname)
plt.close(fig)

