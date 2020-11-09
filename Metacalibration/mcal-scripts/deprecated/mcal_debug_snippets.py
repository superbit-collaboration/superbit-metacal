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

sns.set()

########################################################################
###
###  Run metacal script
### 
########################################################################

python ngmix_fit_superbit2.py meds_files/mock_jitter.meds 400 402 test.asc

python make_annular_catalog.py mock_coadd_cat.ldac mcal-jitter-ngmix2.csv mcal-jitter-ngmix2*asc

# copy to local for plotting
scp jmcclear@ssh.ccv.brown.edu:/users/jmcclear/data/superbit/debug/*.annular /Users/jemcclea/Research/SuperBIT/shear_profiles/debug3


#######################################################################
###
### Run annular on either single image or stack
###
########################################################################


annular -c"x_image y_image g1_nopsf g2_nopsf" -f"nfw_mu >1.01" -s 250 -e 2500 -n 20 truth_5E14shear_300.0_5.fiat 3333 2227 > truth_5e14_nopsf.annular

annular -c"X_IMAGE Y_IMAGE g1_gmix g2_gmix" -s 100 -e 1500 -n 5 mcal-A2218-gmix.fiat 3755 4147  > mcal-A2218-gmixsuperbit

annular -c"X_IMAGE Y_IMAGE g1_Rinv g2_Rinv" -s 120 -e 2200 -n 10 mcal-jitter-ngmix2.fiat 3511 2349 

annular -c"X_IMAGE Y_IMAGE g1 g2" -s 120 -e 2200 -n 12 fitvd-flight-jitter-exp.fiat 3511 2349 > mcal-mocks-gmixsuperbit-Rinv.annular

python annular_jmac.py fitvd-flight-jitter-exp.csv X_IMAGE Y_IMAGE g1 g2


########################################################################
##
##  Debugging T/PSF/etc issues in metacal
##  (this is basically ngmix_fit_superbit, copied over for ipython sesh)
##
########################################################################

medsObj=meds.MEDS('/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-mock/mock_superbit.meds')
index = 1260
psf = medsObj.get_cutout(index,0,type='psf')
im = medsObj.get_cutout(index,0,type='image') 
weight = medsObj.get_cutout(index,0,type='weight')

jj = medsObj.get_jacobian(index,0)
jac = ngmix.Jacobian(row=jj['row0'],col=jj['col0'],dvdrow = jj['dvdrow'],dvdcol=jj['dvdcol'],dudrow=jj['dudrow'],dudcol=jj['dudcol'])

psf_noise = 1e-6
this_psf = psf + 1e-6*np.random.randn(psf.shape[0],psf.shape[1])
this_psf_weight = np.zeros_like(this_psf) + 1./psf_noise**2

psfObs = ngmix.observation.Observation(this_psf,weight = this_psf_weight, jacobian = jac) # can I add a kw for no_pix here???
imageObs = ngmix.observation.Observation(image=im,weight = weight, jacobian = jac, psf = psfObs)

# This contains all 1p/1m 2p/2m noshear information, like im, jac,
# and contains spaces where gmix information will be stored
metaobs = ngmix.metacal.get_all_metacal(imageObs)
mcb=ngmix.bootstrap.MaxMetacalBootstrapper(imageObs)

# This will return sigma, etc. of psf of medsObj
# Then multiply by pixscale & scale to get FWHM
psfim=galsim.Image(psf,scale=0.206)
psfmom = psfim.FindAdaptiveMom()
psfmom.moments_sigma*.206*2.355

# What about when we load the PSF with galsim.des.DES_PSFEx?
# Importing with filename, because galsim knows how to apply WCS
psf_name = '/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/debug3-kron/psfex_output/mockSuperbit_shear_300_1_cat.psf'
im_name = '/Users/jemcclea/Research/GalSim/examples/output-debug/0.4FWHM/mockSuperbit_shear_300_1.fits'
psf_DES = galsim.des.DES_PSFEx(psf_name, im_name) # Do I need to add no_pixel kw here????
psf_DES.sample_scale # yields 0.54230469...
image_pos = galsim.PositionD(2350,2389)   
plt.imshow(psf_DES.getPSFArray(image_pos))
this_psf_des = psf_DES.getPSF(image_pos=image_pos) # returns a whole bunch of attributes; do I need to add no_pixel kw here?
this_psf_des.calculateFWHM() # wait this is STILL too large!!!

# enough screwing around; let's do a maxMetacalBootstrap()
# started by defining _get_priors as in ngmix_fit_superbit.py...
prior = _get_priors()
psf_model='em5'
gal_model='gauss'
ntry=3
Tguess=4*imageObs.jacobian.get_scale()**2
#Tguess=0.169744
metacal_pars={'step': 0.01}
mcb.fit_metacal(psf_model, gal_model, max_pars, Tguess, prior=prior, ntry=ntry,metacal_pars=metacal_pars)
mcr = mcb.get_metacal_result() # this is a dict

########################################################################
##
##  Do size comparisons
##
########################################################################


truth300_1=Table.read('/Users/jemcclea/Research/GalSim/examples/output-debug/truth_shear_300_1.dat',format='ascii')
gals=truth300_1[ (truth300_1['redshift']>0)]
mcal=Table.read('/Users/jemcclea/Research/SuperBIT/metacal/metacal-debug3-shear.csv',format='csv')
full_full = Table.read('/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/debug3/mock_empirical_debug_coadd_cat_full.ldac',format='fits',hdu=2)

gal_match=htm.Matcher(16,ra=gals['ra'],dec=gals['dec'])
all_ind,truth_ind, dist=gal_match.match(ra=mcal['ra'],dec=mcal['dec'],maxmatch=1,radius=8E-4)


plt.plot((gals[truth_ind]['mom_size'])**4,mcal[all_ind]['T_mcal'],'.',label='mcal',alpha=0.5)
plt.plot((gals[truth_ind]['mom_size'])**4,mcal[all_ind]['T_gmix'],'.',label='gmix',alpha=0.5)
#plt.plot(gals[truth_ind]['g1_nopsf'],mcal[all_ind]['g1_Rinv'],'.',label='<R11>^-1')
#plt.plot(gals[truth_ind]['g1_nopsf'],mcal[all_ind]['g1_MC'],'.',label='Projected into R')
plt.xlabel('truth hsm_sigma**4'); plt.ylabel('T size')
plt.xlim(0,20); plt.ylim(-1,10)
plt.legend()


plt.savefig('MetacalT_v_truthMomSize.png')


full_match=htm.Matcher(16,ra=full_full['ALPHAWIN_J2000'],dec=full_full['DELTAWIN_J2000'])
all_ind2,sex_ind, dist=gal_match.match(ra=mcal['ra'],dec=mcal['dec'],maxmatch=1,radius=8E-4)

plt.figure()
plt.plot(full_full[sex_ind]['FLUX_RADIUS'],mcal[all_ind2]['T_mcal'],'.')


plt.plot(full_full[sex_ind]['FLUX_RADIUS'],full_full[sex_ind]['KRON_RADIUS'],'.',alpha=0.5,label='KRON_RADIUS')
plt.plot(full_full[sex_ind]['FLUX_RADIUS'],full_full[sex_ind]['FWHM_IMAGE'],'.',label='FWHM_IMAGE',alpha=0.5)

plt.plot(full_full[sex_ind]['FLUX_RADIUS'],full_full[sex_ind]['FWHM_IMAGE']/2,'.',label='FWHM_IMAGE',alpha=0.5)
plt.plot(full_full[sex_ind]['FLUX_RADIUS'],full_full[sex_ind]['FLUX_RADIUS'],'--k',alpha=0.4,label='FLUX_RADIUS')

#plt.plot(gals[truth_ind]['g1_nopsf'],mcal[all_ind]['g1_MC'],'.',label='Projected into R')
plt.xlabel('FLUX_RADIUS'); plt.ylabel('other size measures')
plt.legend()


#######################################################################
###
### In case I ever want to compare truths against indiv. empirical cats
###
########################################################################

truth300_1=Table.read('/Users/jemcclea/Research/GalSim/examples/output/truth_empiricalPSF_300_1.dat',format='ascii')
gals=truth300_1[truth300_1['redshift']>0]
im300_1 = Table.read('/Users/jemcclea/Research/GalSim/examples/output-safe/mockSuperbit_empiricalPSF_300_1_cat.ldac',format='fits',hdu=2) 
gal_match=htm.Matcher(16,ra=gals['ra'],dec=gals['dec'])

all_ind,truth_ind, dist=gal_match.match(ra=im300_1['ALPHAWIN_J2000'],dec=im300_1['DELTAWIN_J2000'],maxmatch=1,radius=2.5E-4)
truthgals1=gals[truth_ind]; obsgals1=im300_1[all_ind]

obsgals1['truth_g1']=truthgals1['g1_meas']
obsgals1['truth_g2']=truthgals1['g2_meas']
obsgals1['truth_nfw_g1']=truthgals1['nfw_g1']
obsgals1['truth_nfw_g2']=truthgals1['nfw_g2']

obs_e1= (obsgals1['X2_IMAGE']-obsgals1['Y2_IMAGE']) / (obsgals1['X2_IMAGE']+obsgals1['Y2_IMAGE'])
obs_e2= obsgals1['XY_IMAGE']/(obsgals1['X2_IMAGE']+obsgals1['Y2_IMAGE'])

obsgals1['obs_e1']=obs_e1
obsgals1['obs_e2']=obs_e2

xy=np.arange(-0.5,0.5,0.01)

plt.figure()
plt.plot(xy,xy,'-r')
plt.plot(obsgals1['truth_g2'],obs_e2,'.k') 
plt.xlabel('g2_truth'); plt.ylabel('g2_obs')


plt.figure()
plt.plot(xy,xy,'-r')
plt.plot(obsgals1['truth_g1'],obs_e1,'.k') 
plt.xlabel('g1_truth'); plt.ylabel('g1_obs')

