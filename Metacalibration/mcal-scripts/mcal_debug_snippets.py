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

sns.set()

########################################################################
###
###  Run metacal script
### 
########################################################################

python ngmix_fit_testing.py debug3.meds 1000 1010 testfile.asc

python make_annular_catalog.py mock_empirical_debug_coadd_cat.ldac mcal-0.538-PsfScale.csv mcal-0.538PsfScale-fit*asc

# copy to local for plotting
scp jmcclear@ssh.ccv.brown.edu:/users/jmcclear/data/superbit/debug/mcal-0.206-PSFscale.annular /Users/jemcclea/Research/SuperBIT/shear_profiles/debug3


#######################################################################
###
### Run annular on either single image or stack
###
########################################################################


annular -c"X_IMAGE Y_IMAGE g1_meas g2_meas" -f"nfw_mu >1" -s 250 -e 4000 -n 20 truth_shear_300_0.fiat 3333 2227 > truth_5e15_shear.annular

annular -c"X_IMAGE Y_IMAGE g1 g2" -s 250 -e 4000 -n 20 fitvd-debug3.fiat 3511 2349 > fitvd-5e15-shear.annular

annular -c"X_IMAGE Y_IMAGE g1_MC g2_MC" -s 200 -e 4000 -n 20 mcal-0.538-PsfScale.fiat 3511 2349 >  mcal-0.538-PsfScale-MC.annular

annular -c"X_IMAGE Y_IMAGE g1_gmix g2_gmix" -s 200 -e 4000 -n 20 mcal-0.206-PsfScale.fiat 3511 2349  > mcal-0.206-PsfScale-gmix.annular


python annular_jmac.py fitvd-5e15-shear.asc X_IMAGE Y_IMAGE g1 g2 

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

