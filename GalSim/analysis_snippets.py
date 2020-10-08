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
sns.set()

import meds

def get_catalogs(catnames):

    holding={}
    try:
        for i in np.arange(len(catnames)):
            tab=Table.read(catnames[i],format='fits',hdu=2)
            holding["tab{0}".format(i)] = tab
    except:
         for i in np.arange(len(catnames)):
            tab=Table.read(catnames[i],format='ascii')
            holding["tab{0}".format(i)] = tab
       
    all_catalogs=vstack([holding[val] for val in holding.keys()])

    return all_catalogs


def get_stars(truthcats,all_cats):

    """
    isolate stars from GalSim truth catalogs ("truthcats"), and match to SEXtractor
    catalogs ("all_cats"), returning indicies corresponding to stars in all_cats 

    """
    # read in truth catalogs and isolate stars from it (they all have z<=0 by construction!)
    stars=truthcats[truthcats['redshift']<=0]
    
    # Match to observed objects (airy flight cats) in RA/Dec
    star_match=htm.Matcher(16,ra=stars['ra'],dec=stars['dec'])
    all_ind,truth_ind, dist=star_match.match(ra=all_cats['ALPHAWIN_J2000'],dec=all_cats['DELTAWIN_J2000'],maxmatch=1,radius=2.5E-4)

    return all_ind, truth_ind

def get_ellipticities(incat,all_ind,truth_cat,truth_ind):

    e1= (incat['X2_IMAGE']-incat['Y2_IMAGE']) / (incat['X2_IMAGE']+incat['Y2_IMAGE'])
    e2=-2.0*incat['XY_IMAGE']/(incat['X2_IMAGE']+incat['Y2_IMAGE'])
    ellip=np.sqrt(e1**2+e2**2)
    
    e1=e1[all_ind]
    e1.colname='$e_1$'
    e2=e2[all_ind]
    e2.colname='$e_2$'
    ellip=np.sqrt(e1**2+e2**2)
    ellip.colname='$e$'
    ra = incat['ALPHAWIN_J2000'][all_ind]; dec = incat['DELTAWIN_J2000'][all_ind]
    x = incat['X_IMAGE'][all_ind]; y=incat['Y_IMAGE'][all_ind]
    fwhm = incat['FWHM_IMAGE'][all_ind]
    mag = incat['MAG_AUTO'][all_ind]
    flux = incat['FLUX_AUTO'][all_ind]
    flux_rad = incat['FLUX_RADIUS'][all_ind]

    # copy over relevant quantities from truth catalog as well!
    truth_fwhm = truth_cat[truth_ind]['mom_size']*2.355
    truth_flux = truth_cat[truth_ind]['flux']
    truth_g1_meas = truth_cat[truth_ind]['g1_meas']
    truth_g2_meas = truth_cat[truth_ind]['g2_meas']
    
    
    # also add fwhm and mag, useful quantities for stars,

    ellip_tab=Table()
    ellip_tab.add_columns([ra, dec, x, y, e1,e2,ellip,truth_g1_meas,truth_g2_meas,
                               fwhm,flux_rad, mag,flux,truth_fwhm, truth_flux],
                               names=['ra','dec','x', 'y', 'e1','e2','e','g1_meas','g2_meas',
                                         'fwhm','flux_rad','mag','flux','truth_fwhm','truth_flux'])
   

    return ellip_tab

def get_galaxies(truthcats,all_cats):

    """
    isolate galaxies from GalSim truth catalogs ("truthcats"), and match to SEXtractor
    catalogs ("all_cats"), returning indicies corresponding to galaxies in all_cats 

    """
    # read in truth catalogs and isolate stars from it (they all have z<=0 by construction!)
    gals=truthcats[truthcats['redshift']>0]
    
    # Match to observed objects (airy flight cats) in RA/Dec
    gal_match=htm.Matcher(16,ra=gals['ra'],dec=gals['dec'])
    all_ind,truth_ind, dist=gal_match.match(ra=all_cats['ALPHAWIN_J2000'],dec=all_cats['DELTAWIN_J2000'],maxmatch=1,radius=2.5E-4)

    print("Found %d galaies in catalog %s" %(len(all_ind),all_cats))
    return all_ind,truth_ind

def isolate_real_gals(fullcat,starcat):
    try:
        star_matcher=htm.Matcher(16,ra=starcat['ALPHAWIN_J2000'],dec=starcat['DELTAWIN_J2000'])
    except:
        star_matcher=htm.Matcher(16,ra=starcat['ra'],dec=starcat['dec'])
    full_ind, star_ind, dist=star_matcher.match(ra=fullcat['ALPHAWIN_J2000'],dec=fullcat['DELTAWIN_J2000'],maxmatch=1,radius=2.5E-4)
    this_range=np.arange(len(fullcat)) 
    gals=np.setdiff1d(this_range,full_ind)
    gals_only = fullcat[gals]
    #gals_only = gals_only[(gals_only['FWHM_IMAGE']>5) & (gals_only['FLUX_RADIUS']>2.7)] # clean out the junk
    
    return gals_only

###### 300 s exposures ###########

fullcats=glob.glob('/Users/jemcclea/Research/GalSim/examples/output-bandpass/bp_empiricalPSF_300*.ldac')
mock300=get_catalogs(fullcats)

truth300cats=glob.glob('/Users/jemcclea/Research/GalSim/examples/output-bandpass/truth_bp_empiricalPSF_300*dat')
truth300=get_catalogs(truth300cats)
full_ind,stars300_ind=get_stars(truth300,mock300)
stars300=get_ellipticities(mock300,full_ind,truth300,stars300_ind)
gal300_ind,truthgal300_ind=get_galaxies(truth300,mock300) 
gals300=get_ellipticities(mock300,gal300_ind,truth300,truthgal300_ind)

###### 150 s exposures ###########

full150=glob.glob('/Users/jemcclea/Research/GalSim/examples/output-bandpass/bp_empiricalPSF_150*.ldac')
mock150=get_catalogs(full150)

truth150cats=glob.glob('/Users/jemcclea/Research/GalSim/examples/output-bandpass/truth_bp_empiricalPSF_150*dat')
truth150=get_catalogs(truth150cats)
full150_ind,stars150_ind=get_stars(truth150,mock150)
stars150=get_ellipticities(mock150,full150_ind,truth150,stars150_ind)
gal150_ind,truthgal150_ind=get_galaxies(truth150,mock150) 
gals150=get_ellipticities(mock150,gal150_ind,truth150,truthgal150_ind)


########## Real empirical catalogs, as well as mocks ###############

real300cats=glob.glob('/Users/jemcclea/Research/SuperBIT/A2218/Clean/dwb_image*300*WCS_cat.ldac')
real300=get_catalogs(real300cats)
realstar300cats=glob.glob('/Users/jemcclea/Research/SuperBIT/A2218/Clean/*300*WCS_cat.star')
realstar300=get_catalogs(realstar300cats)
realgals300=isolate_real_gals(real300,realstar300)

real150cats=glob.glob('/Users/jemcclea/Research/SuperBIT/A2218/Clean/dwb_image*150*WCS_cat.ldac')
real150=get_catalogs(real150cats)
realstar150cats=glob.glob('/Users/jemcclea/Research/SuperBIT/A2218/Clean/*150*WCS_cat.star')
realstar150=get_catalogs(realstar150cats)
realgals150=isolate_real_gals(real150,realstar150)

#real=Table.read('/Users/jemcclea/Research/SuperBIT_2019/superbit-ngmix/scripts/outputs/A2218_coadd_catalog_full.fits',hdu=2)
real=Table.read('/Users/jemcclea/Research/GalSim/examples/output-superbit/empirical_psfs/v2/A2218_coadd_cat2.fits',hdu=2)
mock=Table.read('/Users/jemcclea/Research/GalSim/examples/output-bandpass/bandpass_empiricalPSF_coadd_full.ldac',hdu=2)  


###################################################################################
########### From Truth catalogs, make plots #######################################
###################################################################################
plt.figure()
plt.hist(gals300['fwhm'],bins=40,alpha=0.5,log=True,label='SExtractor FWHM')
plt.hist(gals300['truth_fwhm'],bins=40,alpha=0.4,log=True,label='Truth (as injected) FWHM')
plt.xlabel('fwhm'); plt.ylabel('number'); plt.legend()
#plt.title('Airy Star Flux Distrib/Empirical PSF/ Noise Added')
#plt.savefig('injected_vs_measured_fwhm_v3.png')  

plt.figure()
plt.hist(gals300['flux'],bins=40,log=True,range=[1,5E5],alpha=0.5,label='SExtractor FLUX_AUTO')
plt.hist(gals300['truth_flux'],bins=40,range=[1,5E5],log=True,alpha=0.5,label='truth (injected) flux')
plt.xlabel('SEX FLUX_AUTO'); plt.ylabel('number'); plt.legend()
#plt.title('Airy Star+Flux Distrib+Empirical PSF+Added Noise')
#plt.savefig('measured_flux_3.png')


###################################################################################
########### Real vs. Mock checkplots        #######################################
###################################################################################


#####################
### 300s checkplots
#####################

# size-mag and flux distributions for real & mock 300s galaxies
plt.figure()
plt.semilogx(realgals300['FLUX_AUTO']/2,realgals300['FWHM_IMAGE'],'.k',label='real gals') 
plt.semilogx(gals300['flux'],gals300['fwhm'],'*',label='mock gals',alpha=0.5)
plt.xlabel('Flux'); plt.ylabel('FWHM'); plt.legend()

plt.figure()
plt.hist(realgals300['FLUX_AUTO'],bins=80,alpha=0.5,range=[1,5E5],log=True, label='real 300s gals')#,density=True) 
plt.hist(gals300['flux'],bins=80,alpha=0.5,range=[1,5E5],log=True,label='mock 300s gals')#,density=True)
plt.xlabel('FLUX_AUTO'); plt.ylabel('log(prob. density)'); plt.legend()

# Check the flux distribution for real & mock 300s full catalogs (not just galaxies)
plt.figure()
plt.hist(real300['FLUX_AUTO'][(real300['FWHM_IMAGE']>5) & (real300['FLUX_RADIUS']>2.7)],range=[1,2E5],bins=80,alpha=0.5,density=True,label='Real 300s FWHM>0.8',log=True)  
plt.hist(mock300['FLUX_AUTO'],range=[1,2E5],bins=80,alpha=0.5,label='Mock 300s',density=True,log=True)
plt.legend()
plt.xlabel('FLUX_AUTO'); plt.ylabel('log(prob)')

# Check the size-mag distributions for real & mock 300s full catalogs (not just galaxies)
plt.figure()
plt.semilogx(real300['FLUX_AUTO'][(real300['FWHM_IMAGE']>5)&(real300['FLUX_RADIUS']>2.7)],
                 real300['FWHM_IMAGE'][(real300['FWHM_IMAGE']>5)&(real300['FLUX_RADIUS']>2.7)],'.k',label='real 300s',alpha=0.5)
plt.semilogx(mock300['FLUX_AUTO']*1.5,mock300['FWHM_IMAGE'],'*b',label='mock 300s',alpha=0.5)
plt.xlabel('FLUX_AUTO'); plt.ylabel('FWHM_IMAGE'); plt.legend(); plt.ylim([-1,60]) 

#####################
### 150s checkplots
#####################

# size-mag and flux distributions for real & mock 150s galaxies
plt.figure()
plt.semilogx(realgals150['FLUX_AUTO'],realgals150['FWHM_IMAGE'],'.k',label='real gals') 
plt.semilogx(gals150['flux'],gals150['fwhm'],'*',label='mock gals',alpha=0.5)
plt.xlabel('Flux'); plt.ylabel('FWHM'); plt.legend()

plt.figure()
plt.hist(real150['FLUX_AUTO'][(real150['FWHM_IMAGE']>5) & (real150['FLUX_RADIUS']>2.7)],bins=80,range=[1,2E5], label='Real 150s cat',log=True,histtype='step')
plt.hist(mock150['FLUX_AUTO']*1.5,bins=80,range=[1,2E5], label='Mock 150s cat',log=True,histtype='step')
plt.legend()
plt.xlabel('FLUX_AUTO'); plt.ylabel('log(prob)')

# Check the flux distribution for real & mock 150s full catalogs (not just galaxies)
plt.figure()
plt.hist(real150['FLUX_AUTO'][(real150['FWHM_IMAGE']>5) & (real150['FLUX_RADIUS']>2.7)],range=[1,2E5],bins=80,alpha=0.5,label='Real 150s FWHM>0.8',log=True)  
plt.hist(mock150['FLUX_AUTO']*1.5,range=[1,2E5],bins=80,alpha=0.5,label='Mock 150s',log=True)
plt.legend()
plt.xlabel('FLUX_AUTO'); plt.ylabel('log(prob)')

# Check the size-mag distributions for real & mock 150s full catalogs (not just galaxies)
plt.figure()
plt.semilogx(real150['FLUX_AUTO'][(real150['FWHM_IMAGE']>5)&(real150['FLUX_RADIUS']>2.7)],
                 real150['FWHM_IMAGE'][(real150['FWHM_IMAGE']>5)&(real150['FLUX_RADIUS']>2.7)],'.k',label='real 150s',alpha=0.5)
plt.semilogx(mock150['FLUX_AUTO']*1.5,mock150['FWHM_IMAGE'],'*b',label='mock 150s',alpha=0.5)
plt.xlabel('FLUX_AUTO'); plt.ylabel('FWHM_IMAGE'); plt.legend(); plt.ylim([-1,60]) 


##########################
### stack image checkplots
##########################

# Do flux & mag histograms for real and mock stacks
# --> not sure of the utility of this, given that star distrib is made up
plt.figure()
plt.hist(real['FLUX_AUTO'][(real['FWHM_IMAGE']>5) & (real['FLUX_RADIUS']>2.7)],bins=80,alpha=0.5,range=[1,1E6], label='Real stack cat',log=True,density=True)
plt.hist(mock['FLUX_AUTO'],bins=80,alpha=0.5,range=[1,1E6], label='Mock stack cat',log=True,density=True)
plt.legend()
plt.xlabel('FLUX_AUTO'); plt.ylabel('log(prob)')

plt.figure()
plt.hist(real['MAG_AUTO'][(real['FWHM_IMAGE']>5) & (real['FLUX_RADIUS']>2.7)],bins=80,range=[12,25], label='Real stack cat',log=True,alpha=0.5,density=True)
plt.hist(mock['MAG_AUTO'],bins=80,range=[12,25], label='Mock stack cat',log=True,alpha=0.5,density=True)
plt.legend()
plt.xlabel('MAG_AUTO'); plt.ylabel('log(prob)')

# 
plt.figure()
plt.semilogx(real['FLUX_AUTO'][(real['FWHM_IMAGE']>5)&(real['FLUX_RADIUS']>2.7)],real['FWHM_IMAGE'][(real['FWHM_IMAGE']>5)&(real['FLUX_RADIUS']>2.7)],'.k',label='real stack',alpha=0.5)
plt.semilogx(mock['FLUX_AUTO'],mock['FWHM_IMAGE'],'*b',label='mock stack',alpha=0.5)
plt.xlabel('FLUX_AUTO'); plt.ylabel('FWHM_IMAGE'); plt.legend(); plt.ylim([-1,60]) 


#####################
### misc. checkplots
#####################


# In case you want to check stars, I guess
plt.figure()
plt.hist(stars150['mag'],bins=100,alpha=0.5,label='Mock 150s stars',density=True)
plt.hist(realstar150['MAG_AUTO'][realstar150['FWHM_IMAGE']>4],range=[12,22],bins=100,alpha=0.5,label='Real 150s stars',density=True)
plt.legend()

# Also in case you want to see mag histograms for different types of images, I guess
plt.figure()
plt.hist(mock150['MAG_AUTO'],range=[12,25],bins=100,alpha=0.5,label='Mock 150s cats',density=True)
plt.hist(mock300['MAG_AUTO'],range=[12,25],bins=100,alpha=0.5,label='Mock 300s cat',density=True)
plt.hist(mock['MAG_AUTO'],range=[12,25],bins=100,alpha=0.5,label='Mock stack cat',density=True)
plt.legend()



###################################################################################
##### In case you want to create a clean galaxy catalog ###########################
###################################################################################

plt.figure()
real300_clean = real300[(real300['FWHM_IMAGE']>4) & (real300['FLUX_RADIUS']>2.7)] # actually, plenty of real objects have FWHM<4, but no stars have FWHM<4 so that's OK
plt.plot(real300_clean['MAG_AUTO'],real300_clean['FWHM_IMAGE'],'.k',alpha=0.8,label='cleaned 300s real catalogs')   

gals300=real300_clean[(real300_clean['FLUX_RADIUS']>= (real300_clean['MAG_AUTO']*-1.942 + 42.3))& (real300_clean['FLUX_RADIUS']>2.7)]
plt.plot(gals300['MAG_AUTO'],gals300['FWHM_IMAGE'],'.r',alpha=0.8,label='gals')

clean_stars_300=real300_clean[(real300_clean['FLUX_RADIUS']< (real300_clean['MAG_AUTO']*-1.942 + 40.5)) & (real300_clean['FLUX_RADIUS']>2.7)]
plt.plot(clean_stars_300['MAG_AUTO'],clean_stars_300['FWHM_IMAGE'],'*k',alpha=0.8,label='stars')


# on empirical stack image
real_clean=full_full[full_full['FWHM_IMAGE']>6]
plt.plot(real_clean['MAG_AUTO'],real_clean['FWHM_IMAGE'],'.b',alpha=0.8,label='cleaned full')   
gals=real_clean[(real_clean['FWHM_IMAGE']>= (real_clean['MAG_AUTO']*-8.98 + 187)) & (real_clean['FLUX_RADIUS']>2.7)
                          & (real_clean['MAG_AUTO']<30)]
plt.plot(gals['MAG_AUTO'],gals['FWHM_IMAGE'],'.r',alpha=0.8,label='gals') 
clean_stars=real_clean[(real_clean['FWHM_IMAGE']< (real_clean['MAG_AUTO']*-8.98 + 187)) & (real_clean['FLUX_RADIUS']>2.7)
                           & (real_clean['MAG_AUTO']<30)]
plt.plot(clean_stars['MAG_AUTO'],clean_stars['FWHM_IMAGE'],'*k',alpha=0.8,label='stars')

#gals150_for_prob = gals150[gals150['MAG_AUTO']>17.8]                                                                                                                     



####################################################################################
########### From histograms, make probability tables  ##############################
####################################################################################

#n,bins=np.histogram(gals['FLUX_AUTO'],bins=100,density=True)#,range=[0,3E6]) --> range is for stars?

n,bins=np.histogram(realstar150['FLUX_AUTO'][realstar150['FWHM_IMAGE']>5],bins=70,density=True,range=[0,1.5E6])
outbins=[]
this=np.arange(len(bins))+1
for i in this[:-1]: 
    m=bins[i-1]*0.5+bins[i]*0.5 
    outbins.append(m)
    
with open('realstar_flux150_prob.txt','w') as f: 
    for l in range(len(outbins)): 
       f.write("%e %.5e \n" %(outbins[l],n[l])) 
f.close()

#### Alternative approach:

from scipy import stats
kde=stats.gaussian_kde(gals['FLUX_AUTO'])
xx=np.linspace(-1,2E7,10000)


##############################################################################################
############### Average together background/noise images, let's see what's happening #########
##############################################################################################

all_bkg_real=glob.glob('/Users/jemcclea/Research/SuperBIT_2019/A2218/Clean/image*300*WCS.bkg.fits')
all_bkg_mock=glob.glob('/Users/jemcclea/Research/GalSim/examples/output/empiricalPSF*300*.bkg.fits')

BkgRealArr = []
for bkg in all_bkg_real:
    this_bkg = fits.getdata(bkg)
    BkgRealArr.append(this_bkg)
masterBkgReal=np.median(BkgRealArr,axis=0)
hdu1=fits.PrimaryHDU()
hdu1.data=masterBkgReal
hdu1.writeto('real300s_bkg.fits')
plt.imshow(masterBkgReal)

BkgMockArr = []
for bkg in all_bkg_mock:
    this_bkg = fits.getdata(bkg)
    BkgMockArr.append(this_bkg)
masterBkgMock=np.median(BkgMockArr,axis=0)
hdu1=fits.PrimaryHDU()
hdu1.data=masterBkgMock
hdu1.writeto('mock300s_bkg.fits')

    
##############################################################################################
########### Make some star FWHM histograms ##########################################
##############################################################################################

sns.set(style="ticks", color_codes=True)
f1,ax1=plt.subplots(figsize=[10,8])

sns.distplot(stars300['fwhm']*.206,bins=30,label=(r'stars 300 $\langle fwhm^{\star} \rangle $ = %.4f"'%np.median(stars300['fwhm']*.206)),
                 norm_hist=True,kde=False,hist_kws=hist_keys)
sns.distplot(realstar300['FWHM_IMAGE']*.206,bins=30,label=(r'realstars 300 $\langle fwhm^{\star} \rangle $ = %.4f"'%np.median(realstar300['FWHM_IMAGE']*.206)),
                 norm_hist=True,kde=False,hist_kws=hist_keys)
sns.distplot(stars150['fwhm']*.206,bins=30,label=(r'stars 150 $\langle fwhm^{\star} \rangle $ = %.4f"'%np.median(stars150['fwhm']*.206)),
                 norm_hist=True,kde=False,hist_kws=hist_keys)
sns.distplot(realstar150['FWHM_IMAGE']*.206,bins=30,label=(r'realstars 150 $\langle fwhm^{\star} \rangle $ = %.4f"'%np.median(realstar150['FWHM_IMAGE']*.206)),
                 norm_hist=True,kde=False,hist_kws=hist_keys)

ax1.set_xlabel('fwhm of stars')
ax1.set_ylabel('norm. histogram')
ax1.legend()

##############################################################################################
########### Run SEXtractor on coadd without filtering out non-analysis objects ###############
##############################################################################################


sex mock_empirical_debug_coadd.fits -WEIGHT_IMAGE mock_empirical_debug_coadd.weight.fits -CATALOG_NAME mock_empirical_debug_coadd_full.ldac -PARAMETERS_NAME /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/sextractor.param -STARNNW_NAME /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/default.nnw -FILTER_NAME /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/default.conv -c /Users/jemcclea/Research/SuperBIT/superbit-ngmix/superbit/astro_config/sextractor.config


##############################################################################################
########### If you want to filter out stars from mock catalogs  ###############
##############################################################################################

fulln='/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-empirical/mock_empirical_psf_coadd_cat_full.ldac'
full=Table.read(fulln,format='fits',hdu=2)

truthdir = '/Users/jemcclea/Research/GalSim/examples/output-deep-gauss/'
#truthdir = '/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-jitters'
truthcatn = 'truth_0.3FWHM_gaussStar_300_5.dat'

truthfile=os.path.join(truthdir,truthcatn)
truthcat = Table.read(truthfile,format='ascii')
stars=truthcat[truthcat['redshift']==0] 

star_matcher = htm.Matcher(16,ra=stars['ra'],dec=stars['dec'])
matches,starmatches,dist = star_matcher.match(ra=full['ALPHAWIN_J2000'],
                                                    dec=full['DELTAWIN_J2000'],radius=5E-4,maxmatch=1)

stars = full[matches]

# Save result to file, return filename
outname = fulln.replace('.ldac','.star')
ffull[fmatches].write(outname,format='fits',overwrite=True)


# Make cute plots, if desired:

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

label = (r'$\langle fwhm^{\star} \rangle $ = %.4f'%np.median(stars['FWHM_WORLD']*3600))
plt.hist(stars['FWHM_WORLD']*3600, bins=50,label=label)
plt.xlabel('stellar FWHM (arseconds)')


fig, axs = plt.subplots(1, 1, tight_layout=True)
# N is the count in each bin, bins is the lower-limit of the bin
N, bins, patches = axs.hist(stars['FWHM_WORLD']*3600, bins=50,label=label)

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())

# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
fig.savefig('stellar_fwhm.png')

##############################################################################################
########### Making new fitvd catalogs, with X/Y info, for annular shear profiles #############
##############################################################################################

## In case I can't find it:
fitvd --seed 192308545 --config ../fitvd-superbit-exp.yaml --output fitvd-flight-jitter-exp.fit mock_jitter.meds

fitvd=Table.read('output-jitter/fitvd-flight-jitter-exp.fit',format='fits',hdu=1)
gals=Table.read('output-jitter/mock_coadd_cat.ldac',format='fits',hdu=2) #contains only analysis objects.
full=Table.read('output-jitter/mock_coadd_cat_full.ldac',format='fits',hdu=2)

plt.plot(full['MAG_AUTO'],full['FWHM_IMAGE'],'.k',alpha=0.4,label='all objects') 
plt.plot(gals['MAG_AUTO'],gals['FWHM_IMAGE'],'.r',alpha=0.4,label='analysis objects')

plt.plot(stars['MAG_AUTO'],stars['FWHM_IMAGE'],'*b',alpha=0.4,label='stars') 


#cleangals=full[(full['FWHM_IMAGE']>= (full['MAG_AUTO']*-9.968 + 212))]

fitvd_matcher = htm.Matcher(16,ra=fitvd['ra'],dec=fitvd['dec'])
gals_ind,fitvd_ind,dist=fitvd_matcher.match(ra=gals['ALPHAWIN_J2000'],dec=gals['DELTAWIN_J2000'],radius=5E-4,maxmatch=1)
fitvd=fitvd[fitvd_ind]; gals=gals[gals_ind]

newtab=Table()
newtab.add_columns([fitvd['id'],fitvd['ra'],fitvd['dec'],fitvd['exp_g'][:,0],fitvd['exp_g'][:,1]],names=['id','ra','dec','g1','g2'])
newtab.add_columns([fitvd['exp_T'],['exp_flux']],names=['T','flux'])
newtab.add_columns([gals['X_IMAGE'],gals['Y_IMAGE']])  
newtab.write('output-jitter/fitvd-flight-jitter-exp.csv',format='csv',overwrite=True) 


## Convert to fiatformat:
cmd='sdsscsv2fiat output-jitter/fitvd-flight-jitter-exp.csv > output-jitter/fitvd-flight-jitter-exp.fiat'
os.system(cmd)

## On command line 
annular -c"X_IMAGE Y_IMAGE g1 g2" -f "g1>-2" -s 150 -e 2500 -n 8 fitvd-real.fiat 3511 2349  > fitvd-bit.annular
annular -c"X_IMAGE Y_IMAGE g1 g2" -f "g1>-2" -s 100 -e 1500 -n 5 fitvd-A2218.fiat 3736 4149    > fitvd-bit.annular
