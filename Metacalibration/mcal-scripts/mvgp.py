from astropy.table import vstack, Table
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from astropy.io import fits
import glob
from esutil import htm
import sys
import os
import math
import galsim
import galsim.des
import pdb
import scipy
import seaborn as sns
import meds
import ngmix


"""
Snippets to generate a minimum viable galsim product, and make sure that responsivity
is what wre would expect

"""

###
### Start by making a Gaussian galaxy, and convolving with a small PSF
###

gal_flux = 1.e5    # total counts on the image
gal_fwhm = 0.1     # arcsec
psf_fwhm = 0.2     # arcsec
pixel_scale = 0.02  # arcsec / pixel

gal = galsim.Gaussian(flux=gal_flux, fwhm=gal_fwhm)
psf = galsim.Gaussian(flux=1., fwhm=psf_fwhm)
final = galsim.Convolve([gal, psf])

stamp = final.drawImage(scale=pixel_scale)
plt.imshow(stamp.array) 

### 
### Shear galaxy by 2% in each direction, and convolve with a PSF
###

gal_shape = galsim.Shear(g1=0.02, g2 = 0.02)
shear = gal.shear(gal_shape)

shearedgal = galsim.Convolve([shear,psf])
shearim = shearedgal.drawImage(scale=pixel_scale)

shearedgal.calculateFWHM() # result is 0.22344066 arcsec
sigma = shearim.FindAdaptiveMom().moments_sigma # result is 4.75723981857299 pix, equiv. to T= 45.26 pix or fwhm = 2.289 arcsec
shape = shearim.FindAdaptiveMom().observed_shape # result is 0.0039872733202036435+0.00398729334427568j

###
### Ok, now start convolving galaxy with 1p/1m/2p/2m
###
###       1p -> ( shear, 0)
###       1m -> (-shear, 0)
###       2p -> ( 0, shear)
###       2m -> ( 0, -shear)

shear_step = 0.01

# Let's assume we know PSF precisely, and have deconvolved the image exactly with a very slightly larger PSF

psf_enlarged = galsim.Gaussian(flux=1., fwhm=psf_fwhm*1.05)
noshear = galsim.Convolve([gal, psf])

noshear_im = noshear.drawImage(scale = pixel_scale) # not sure if scale is the right thing to set...
noshear_sigma = noshear_im.FindAdaptiveMom().moments_sigma
noshear_shape = noshear_im.FindAdaptiveMom().observed_shape

# 1p/1m 

shear_1p = galsim.Shear(g1=shear_step, g2 = 0.0)
shear_1m = galsim.Shear(g1=-1*shear_step, g2 = 0.0)
gal_1p = noshear.shear(shear_1p)
gal_1m = noshear.shear(shear_1m)

gal1p_im = gal_1p.drawImage(scale=pixel_scale)
gal1m_im = gal_1m.drawImage(scale=pixel_scale)

sigma_1p = gal1p_im.FindAdaptiveMom().moments_sigma
shape_1p = gal1p_im.FindAdaptiveMom().observed_shape

sigma_1m = gal1m_im.FindAdaptiveMom().moments_sigma
shape_1m = gal1m_im.FindAdaptiveMom().observed_shape

# 2p/2m

shear_2p = galsim.Shear(g1=shear_step, g2 = 0.0)
shear_2m = galsim.Shear(g1=-1*shear_step, g2 = 0.0)
gal_2p = noshear.shear(shear_2p)
gal_2m = noshear.shear(shear_2m)

gal2p_im = gal_2p.drawImage(scale=pixel_scale)
gal2m_im = gal_2m.drawImage(scale=pixel_scale)

sigma_2p = gal2p_im.FindAdaptiveMom().moments_sigma
shape_2p = gal2p_im.FindAdaptiveMom().observed_shape

sigma_2m = gal2m_im.FindAdaptiveMom().moments_sigma
shape_2m = gal2m_im.FindAdaptiveMom().observed_shape



