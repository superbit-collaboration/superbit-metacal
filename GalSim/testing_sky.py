import numpy as np
import matplotlib.pyplot as plt
plt.ion()

import sys
import os
import galsim
import galsim.des
import pdb
import scipy
import ngmix
import meds
import mof
from numpy import random
from mpl_toolkits.axes_grid1 import ImageGrid

"""
 - Purpose of this code is to... find the place at which the breakdown of galaxy detection occurs in b filter

 - What am I trying to prove?  Which sky background matches expection that a mag=19 galaxy can  
   be detected on a single 300s exposure, while a mag=24 galaxy can be detected in 4000s of exposure
"""

###
### Set some parameters
###

# define an inclinedSersic galaxy

gal_flux = 4844.7                    # flux of m=19.5 galaxy observed for 300 s
half_light_radius = 0.6094           # np.mean(cosmos_cat['hlr_cosmos10']*0.03)
n = 2.479                            # cosmos_cat['n_sersic_cosmos10'][index]
scale_h_over_r = 0.53                # np.mean(cosmos['q_cosmos10'])
inclination = (np.pi - 0.202)*galsim.radians  # np.mean(cosmos['phi_cosmos10']) = 0.202

# Characterizing IMX455

read_noise = 2                       # e-
gain = 0.81                          # e-/ADU, which tells GalSim what to do with RN in e-.
pixel_scale = 0.144                  # arcsec / pixel
psf_fwhm = 0.33                      # arcsec
base_sky_level = 0.12               # ADU / s / pix; 0.048 = Ajay estim., 0.12 = Shaaban estim.

# Characterizing observations

n_obs = 14
exp_time = 300                           # s
sky_level =  base_sky_level * exp_time   # ADU / pix
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
psf_stamp = psf.drawImage(scale=psf_pixel_scale) 
obs_stamp = final.drawImage(scale=pixel_scale,nx=50,ny=50)

## Inspect single galaxy observation before noise

plt.figure()
plt.imshow(obs_stamp.array)
plt.title('PSF-convolved galaxy\n%d s observation (noise-free)' % int(exp_time))
plt.savefig('diagnostics_plots/m19.5_noisefree_gal.png')

## Create noisy observations
image_obslist = [] 
for n in range(n_obs):
    ##
    ## Define noise, add it to stamps
    ##
    this_gal_stamp = final.drawImage(scale=pixel_scale,nx=50,ny=50)
    ud = galsim.UniformDeviate(seed+n+1)
    noise = galsim.CCDNoise(rng=ud,sky_level=sky_level,gain=1/gain,read_noise=read_noise)
    this_gal_stamp.addNoise(noise)
    image_cutout = this_gal_stamp.array
    image_obslist.append(image_cutout)
    del(this_gal_stamp)

## Inspect a single noisy observation
plt.figure()
plt.imshow(image_cutout)
plt.title('Single galaxy observation \nbase sky bkg = %.3f exptime = %d s' % (base_sky_level,int(exp_time)))
plt.colorbar()
plt.savefig('diagnostics_plots/brightgal_single_noisyObs2.png')

# See what coadded image might look like
junk_avg = np.median(image_obslist,axis=0)
plt.figure()
plt.imshow(junk_avg)
plt.title('Noisy galaxy nexp=%d "coadd"\nbase sky bkg = %.3f exptime = %d s' % (n_obs,base_sky_level,int(exp_time)))
plt.savefig('diagnostics_plots/brightgal_medianCombine_noisyObs2.png')

#####################################################
#####################################################

# Set up figure and image grid
fig = plt.figure(figsize=(21, 10))
grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(2,3),
                 axes_pad=1.0,
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
ax4=axes[3];ax5=axes[4];ax6=axes[5]

image_cutout = junk_avg

fl1 = gal_stamp.FindAdaptiveMom().moments_sigma*pixel_scale
fl2=gal_stamp.FindAdaptiveMom().observed_shape.g1; fl3=gal_stamp.FindAdaptiveMom().observed_shape.g2
ax1.imshow(image_cutout,vmin=0,vmax=(image_cutout.max()*.90))
ax1.set_title('obs shear = [%.3e,%.3e]\nobs sigmaMom = %.3e' % (fl1,fl2,fl3),fontsize=10)

[fl1,fl2]=res['g']; fl3=(np.sqrt(res['T']/2))
x,y=gal_gm.get_cen()
ax2.imshow(gal_model_im,vmin=0,vmax=(image_cutout.max()*.90))
ax2.set_title('psf_pixscale=0.206" model\nshear  = [%.3e,%.3e]\nmodel sigma = %.3e' % (fl1,fl2,fl3),fontsize=10)
ax2.plot((x+image_cutout.shape[0]/2),(y+image_cutout.shape[1]/2),'pr')

resi = ax3.imshow(gal_model_im-image_cutout,vmin=-10,vmax=10)
ax3.set_title('psf_pixscale=0.206" residuals',fontsize=10)
ax3.cax.colorbar(resi)
#ax3.cax.toggle_label(True)


fl1 = gal_stamp.FindAdaptiveMom().moments_sigma*pixel_scale
fl2=gal_stamp.FindAdaptiveMom().observed_shape.g1; fl3=gal_stamp.FindAdaptiveMom().observed_shape.g2
ax4.imshow(junk_avg,vmin=0,vmax=(image_cutout.max()*.90))
ax4.set_title('obs shear = [%.3e,%.3e]\nobs sigmaMom = %.3e' % (fl1,fl2,fl3),fontsize=10)

[fl1,fl2]=res_realsigma['g']; fl3=(np.sqrt(res_realsigma['T']/2))
x,y=gal_gm2.get_cen()
print("realSigma model x=%.3f,y=%.3f" % (x,y))
ax5.imshow(gal_model_im2,vmin=0,vmax=(image_cutout.max()*.90))
ax5.plot((x+image_cutout.shape[0]/2),(y+image_cutout.shape[1]/2),'pr')
ax5.set_title('psf_pixscale=0.05"\nshear = [%.3e,%.3e]\nsigmaMom = %.3e' % (fl1,fl2,fl3),fontsize=10)

resi2 = ax6.imshow(gal_model_im2-image_cutout,vmin=-10,vmax=10)
ax6.set_title('psf_pixscale=0.05" residuals',fontsize=10)
ax6.cax.colorbar(resi2)
#ax3.cax.toggle_label(True)

fl1 = sheared_stamp.FindAdaptiveMom().moments_sigma*pixel_scale
fig.suptitle('%e flux/%f" FWHM galaxy + real PSF + 0.206" plate scale\napplied shear = [-0.02,0.02] real sigmaMom = %.3e'
                 % (gal_flux,gal_fwhm,fl1))

fig.savefig('GalModel_diagnostic_psfPixscales_em3_gal3.png')
plt.close(fig)


####
#### Favorite comparison tool: how would this look in a coadd?
####
"""
junk_im = []
for i in range(n_obs): 
    junk_im.append(image_obslist[i].image) 
junk_avg = np.mean(junk_im,axis=0)

"""
