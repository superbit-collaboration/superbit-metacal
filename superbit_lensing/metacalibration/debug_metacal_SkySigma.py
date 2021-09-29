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
Purpose of this code is to... find the place at which the breakdown of ngmix fitting occurs
 - This requires making known galaxy, *with known size* and known PSF(also with *known size*
 - We will start by having a nearly noiseless image with an over-resolved PSF.
 - Then gradually increase noise, while allowing the number of observations to increase 
 - Then then, increase pixel scale
 - What am I trying to prove?  
      (a) When does recovered size get really big or small (a/k/a no more shear) 
      (b) When does recovered shear get too large or small
"""


###
### Set some parameters that will get called
###
gal_flux = 5.e2    # total counts on the image
gal_fwhm = 1     # arcsec
psf_fwhm = 0.39     # arcsec
pixel_scale = 0.206  # arcsec / pixel
n_obs = 25
psf_noise = 1e-6
sky_sigma = (0.057*300)**2
sky_level = 31 # ADU/pix 0.57 = debug simul., 1.826 = real from Ajay paper 
read_noise = 2 # e-, this is quite small compared to real CCD 
gain = 1/3.333 # e-/ADU, which tells GalSim what to do with RN in e-.
#seed = 75748946
seed = random.randint(11111111,99999999)

# define priors or whatever
# mcal_shear = 0.01
lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
max_pars = {'method':'lm','lm_pars':lm_pars,'find_center':True}

psf_model = 'em3' 
gal_model = 'gauss'


def _get_priors():

        # prior on ellipticity.  The details don't matter, as long
        # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014
        
        g_sigma = 0.3
        g_prior = ngmix.priors.GPriorBA(g_sigma)
        
        # 2-d gaussian prior on the center
        # row and column center (relative to the center of the jacobian, which would be zero)
        # and the sigma of the gaussians
        
        # units same as jacobian, probably arcsec
        row, col = 0.0, 0.0
        row_sigma, col_sigma = 0.2, 0.2 
        cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma)
        
        # T prior.  This one is flat, but another uninformative you might
        # try is the two-sided error function (TwoSidedErf)
        
        Tminval = -0.1 # arcsec squared
        Tmaxval = 2000
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval)
        
        # similar for flux.  Make sure the bounds make sense for
        # your images
        
        Fminval = -1.e1
        Fmaxval = 1.e5
        F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval)
        
        # now make a joint prior.  This one takes priors
        # for each parameter separately
        priors = ngmix.joint_prior.PriorSimpleSep(
        cen_prior,
        g_prior,
        T_prior,
        F_prior)
    
        return priors


def _make_psf():
    jitter_fwhm = 0.3
    jitter = galsim.Gaussian(flux=1., fwhm=jitter_fwhm)
    
    lam_over_diam = 0.257831 # units of arcsec
    aberrations = np.zeros(38)             # Set the initial size.
    aberrations[0] = 0.                       # First entry must be zero                       
    aberrations[1] = -0.00305127
    aberrations[4] = -0.02474205              # Noll index 4 = Defocus
    aberrations[11] = -0.01544329             # Noll index 11 = Spherical
    aberrations[22] = 0.00199235
    aberrations[26] = 0.00000017
    aberrations[37] = 0.00000004
    
    optics = galsim.OpticalPSF(lam=625,diam=0.5,
                                   obscuration=0.38, nstruts=4,
                                   strut_angle=(90*galsim.degrees), strut_thick=0.087,
                                   aberrations=aberrations)
    
    psf = galsim.Convolve([jitter,optics])
    return psf

prior=_get_priors()
gpsf = _make_psf()

######################################
## Case 1: psf_pixel_scale = 0.05
######################################

psf_pixel_scale = 0.05

###      
### Now draw galaxy and PSF
gal = galsim.Gaussian(flux=gal_flux, fwhm=gal_fwhm)
gal_shape = galsim.Shear(g1=-0.02, g2 = 0.02)
sheared = gal.shear(gal_shape)
#psf = galsim.Gaussian(flux=1., fwhm=psf_fwhm)
psf = gpsf
final = galsim.Convolve([sheared, psf])

gal_only_stamp = gal.drawImage(scale=pixel_scale)
sheared_stamp = sheared.drawImage(scale=pixel_scale)
psf_stamp = psf.drawImage(scale=psf_pixel_scale,nx=256,ny=256) 
gal_stamp = final.drawImage(scale=pixel_scale,nx=32,ny=32)


###
### Do fits with the real sky sigma and this pixel scale
###


image_obslist_realsigma = ngmix.observation.ObsList()

for n in range(n_obs):
    ##
    ## Define noise, add it to stamps
    ##

    # limit size for over-resolved PSF
    this_gal_stamp = final.drawImage(scale=pixel_scale,nx=32,ny=32)
    ud = galsim.UniformDeviate(seed+n+1)
    noise = galsim.CCDNoise(rng=ud,sky_level=sky_level, gain=gain,read_noise=read_noise)
    this_gal_stamp.addNoise(noise)

    ##
    ## Set up for ngmix fitting
    ##

    psf_cutout = psf_stamp.array
    image_cutout = this_gal_stamp.array

    jj_psf = ngmix.jacobian.DiagonalJacobian(scale=psf_pixel_scale,x=psf_cutout.shape[0]/2,y=psf_cutout.shape[1]/2)
    jj_im = ngmix.jacobian.DiagonalJacobian(scale=pixel_scale,x=image_cutout.shape[0]/2,y=image_cutout.shape[1]/2)

    # Make PSF ngmix.Observation(); apparently PSF fitter needs a teensy bit of noise
    this_psf = psf_cutout + psf_noise*np.random.randn(psf_cutout.shape[0],psf_cutout.shape[1])
    this_psf_weight = np.zeros_like(this_psf) + 1./psf_noise**2
    psfObs = ngmix.observation.Observation(psf_cutout,weight = this_psf_weight, jacobian = jj_psf)

    # Make image ngmix.Observation()
    this_weight = np.zeros_like(image_cutout)+ 1./sky_sigma
    imageObs = ngmix.observation.Observation(image_cutout,weight = this_weight,jacobian = jj_im, psf = psfObs)
    
    image_obslist_realsigma.append(imageObs)
    del(this_gal_stamp,imageObs)

# What does it look like to you?
plt.imshow(image_cutout)

####
#### do ngmix fitting
####

mcal_obs_realsigma = ngmix.metacal.get_all_metacal(image_obslist_realsigma)

# just do no_shear fit
boot_realsigma = ngmix.Bootstrapper(mcal_obs_realsigma['noshear'])
boot_realsigma.fit_psfs(psf_model,1.)

# Alright, now let's see the actual galaxy fit
# wake up. This means... running the boot fit, and then plotting up that galaxy
boot_realsigma.fit_max(gal_model,max_pars,prior=prior)
res_realsigma = boot_realsigma.get_fitter().get_result()

# Wanna see the galaxy model get rendered?
junk_im = []
for i in range(n_obs): 
    junk_im.append(image_obslist_realsigma[i].image) 
junk_avg = np.mean(junk_im,axis=0)

gal_gm2 = boot_realsigma.get_fitter().get_convolved_gmix()
gal_model_im2 = gal_gm2.make_image(image_cutout.shape,jacobian=jj_im)
"""
fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
ax1.imshow(junk_avg)#,vmin=0,vmax=(psf_cutout.max()*.90))
ax2.imshow(gal_model_im2)#,vmin=0,vmax=(psf_cutout.max()*.90))
ax3.imshow(junk_avg-gal_model_im2)
"""
t_arr=[]
for p in boot_realsigma.mb_obs_list[0]: 
    print(p.psf.get_gmix().get_e1e2T()) 
    t_arr.append(p.psf.get_gmix().get_e1e2T()[2])   

mean_T = np.mean(t_arr)
mean_sigma = np.sqrt(mean_T/2)


# Wanna see the PSF model rendered??
psf_gm = p.psf.get_gmix()
psf_model_im = psf_gm.make_image(psf_cutout.shape,jacobian=jj_psf)
fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
ax1.imshow(psf_cutout)#,vmin=0,vmax=(psf_cutout.max()*.90))
ax2.imshow(psf_model_im)#,vmin=0,vmax=(psf_cutout.max()*.90))
ax3.imshow(psf_cutout-psf_model_im)
fig.suptitle('0.39" FWHM real PSF + em3 fot + %.2f" psf_pixscale' % psf_pixel_scale)
fig.savefig('psf_diagnostic_0.39FWHM_%.2fPSFpxscl_em3.png' % psf_pixel_scale)
plt.close(fig)


# Check goodness of PSF fit
print("\nThis is PSF model for pixscale = %f\n" % psf_pixel_scale)
print(psf_stamp.FindAdaptiveMom().observed_shape) # yields O(1E-9) shear, as expected
print(psf_stamp.calculateFWHM()) # yields 0.22583, also as expected
print(psf_stamp.FindAdaptiveMom().moments_sigma*psf_pixel_scale) # yields 0.095123
#print(psf_gm.get_g1g2sigma())
print("model PSF sigma = %f" % mean_sigma)


######################################
## Case 2: psf_pixel_scale = 0.206
######################################

del(psf_pixel_scale)
psf_pixel_scale = 0.206

###      
### Now draw galaxy and PSF
gal = galsim.Gaussian(flux=gal_flux, fwhm=gal_fwhm)
gal_shape = galsim.Shear(g1=-0.02, g2 = 0.02)
sheared = gal.shear(gal_shape)
psf = gpsf
final = galsim.Convolve([sheared, psf])

gal_only_stamp = gal.drawImage(scale=pixel_scale)
sheared_stamp = sheared.drawImage(scale=pixel_scale)
psf_stamp = psf.drawImage(scale=psf_pixel_scale,nx=16,ny=16) 
gal_stamp = final.drawImage(scale=pixel_scale,nx=16,ny=16)


#
# Do fits with the real sky sigma and this pixel scale
#
#


# Create obs_list for pixscale2

image_obslist_pixscale2 = ngmix.observation.ObsList()

for n in range(n_obs):
    ##
    ## Define noise, add it to stamps
    ##

    this_gal_stamp = final.drawImage(scale=pixel_scale,nx=32,ny=32)
    ud = galsim.UniformDeviate(seed+n+1)
    noise = galsim.CCDNoise(rng=ud,sky_level=sky_level, gain=1/gain,read_noise=read_noise)
    this_gal_stamp.addNoise(noise)

    ##
    ## Set up for ngmix fitting
    ##

    psf_cutout = psf_stamp.array
    image_cutout = this_gal_stamp.array
    #image_cutout[image_cutout <=0] = 1E-2 #--> never do this

    jj_psf = ngmix.jacobian.DiagonalJacobian(scale=psf_pixel_scale,x=psf_cutout.shape[0]/2,y=psf_cutout.shape[1]/2)
    jj_im = ngmix.jacobian.DiagonalJacobian(scale=pixel_scale,x=image_cutout.shape[0]/2,y=image_cutout.shape[1]/2)

    # Make PSF ngmix.Observation(); apparently PSF fitter needs a teensy bit of noise
    this_psf = psf_cutout + psf_noise*np.random.randn(psf_cutout.shape[0],psf_cutout.shape[1])
    this_psf_weight = np.zeros_like(this_psf) + 1./psf_noise**2
    psfObs = ngmix.observation.Observation(psf_cutout,weight = this_psf_weight, jacobian = jj_psf)

    # Make image ngmix.Observation()
    this_weight = np.zeros_like(image_cutout)+ 1./sky_sigma
    imageObs = ngmix.observation.Observation(image_cutout,weight = this_weight,jacobian = jj_im, psf = psfObs)
    
    image_obslist_pixscale2.append(imageObs)
    del(this_gal_stamp,imageObs)

####
#### do ngmix fitting
####

mcal_obs = ngmix.metacal.get_all_metacal(image_obslist_pixscale2)

# just do no_shear fit
boot = ngmix.Bootstrapper(mcal_obs['noshear'])
boot.fit_psfs(psf_model,1.)

# Alright, now let's see the actual galaxy fit
# wake up. This means... running the boot fit, and then plotting up that galaxy
boot.fit_max(gal_model,max_pars,prior=prior)
res = boot.get_fitter().get_result()

# Wanna see THIS PSF model get rendered?
# Wanna see the PSF model rendered??
p = boot.mb_obs_list[0][1]
psf_gm = p.psf.get_gmix()
psf_model_im = psf_gm.make_image(psf_cutout.shape,jacobian=jj_psf)
fig,(ax1,ax2,ax3) = plt.subplots(nrows=1,ncols=3,figsize=(21,7))
ax1.imshow(psf_cutout,vmin=0,vmax=(psf_cutout.max()*.90))
ax2.imshow(psf_model_im,vmin=0,vmax=(psf_cutout.max()*.90))
ax3.imshow(psf_cutout-psf_model_im)
fig.suptitle('0.39" FWHM real PSF + em3 fit + %.2f" psf_pixscale' % psf_pixel_scale)
fig.savefig('psf_diagnostic_0.39FWHM_%.2fPSFpxscl_em3.png' % psf_pixel_scale)
plt.close(fig)

t_arr=[]
for p in boot.mb_obs_list[0]: 
    #print(p.psf.get_gmix().get_e1e2T()) 
    t_arr.append(p.psf.get_gmix().get_e1e2T()[2])   

mean_T = np.mean(t_arr)
mean_sigma = np.sqrt(mean_T/2)


# Check goodness of PSF fit
print("\nThis is for pixscale = %f\n" % psf_pixel_scale)
print(psf_stamp.FindAdaptiveMom().observed_shape) # yields O(1E-9) shear, as expected
print(psf_stamp.calculateFWHM()) # yields 0.22583, also as expected
print(psf_stamp.FindAdaptiveMom().moments_sigma*psf_pixel_scale) 
#print(psf_gm.get_g1g2sigma())
print("model PSF sigma = %f\n" % mean_sigma)

# Wanna see the galaxy model get rendered?
gal_gm = boot.get_fitter().get_convolved_gmix()
gal_model_im = gal_gm.make_image(image_cutout.shape,jacobian=jj_im)

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


###
### Now do comparisons
###

# Sheared Galaxy #
print("\nsheared galaxy shape, FWHM & sigma")
print(sheared_stamp.FindAdaptiveMom().observed_shape) # yields -2.774e-09+6.233e-09j
print(sheared_stamp.calculateFWHM()) # yields 0.1051, which is weird because that's 5% bigger than inject
print(sheared_stamp.FindAdaptiveMom().moments_sigma*pixel_scale) # yields 0.04285890

# PSF convolved Galaxy #
print("\nPSF-convolved galaxy shape, FWHM & sigma")
print(gal_stamp.FindAdaptiveMom().observed_shape) # yields O(1E-9) shear, as expected
print(gal_stamp.calculateFWHM()) # yields 0.22583, also as expected
print(gal_stamp.FindAdaptiveMom().moments_sigma*pixel_scale) # yields 0.095123

# Fit results #
print("\nNGMixshape & sigma for psf_pixscale=%f"%psf_pixel_scale)
print(res['g']) 
print(np.sqrt(res['T']/2)) # yields 0.0386337, but this varies



####
#### Favorite comparison tool: how would this look in a coadd?
####
"""
junk_im = []
for i in range(n_obs): 
    junk_im.append(image_obslist[i].image) 
junk_avg = np.mean(junk_im,axis=0)

"""
