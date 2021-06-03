from astropy.table import vstack, hstack, Table
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
from astropy.io import fits
import glob
from esutil import htm
import sys
import os 
import astropy

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
    stars=truthcats[truthcats['redshift']==0]
    # Match to observed objects (airy flight cats) in RA/Dec
    star_match=htm.Matcher(16,ra=stars['ra'],dec=stars['dec'])
    all_ind,stars_ind, dist=star_match.match(ra=all_cats['ALPHAWIN_J2000'],dec=all_cats['DELTAWIN_J2000'],maxmatch=1,radius=1.5E-4)
    master_star_cat=hstack([all_cats[all_ind],stars[stars_ind]])
    
    return master_star_cat

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
    z =truth_cat[truth_ind]['redshift']
    
    
    # also add fwhm and mag, useful quantities for stars,

    ellip_tab=Table()
    ellip_tab.add_columns([ra, dec, x, y, e1,e2,ellip,truth_g1_meas,truth_g2_meas,
                               fwhm,flux_rad, mag,flux,truth_fwhm, truth_flux,z],
                               names=['ra','dec','x', 'y', 'e1','e2','e','g1_meas','g2_meas',
                                         'fwhm','flux_rad','mag','flux','truth_fwhm','truth_flux','redshift'])
   

    return ellip_tab

def get_galaxies(truthcats,all_cats):

    """
    isolate galaxies from GalSim truth catalogs ("truthcats"), and match to SEXtractor
    catalogs ("all_cats"), returning indicies corresponding to galaxies in all_cats 

    """
    # read in truth catalogs and isolate stars from it (they all have z<=0 by construction!)
    truthgals=truthcats[truthcats['redshift']>0]
    
    # print("a total of %d gals were injected into simulations"%len(gals))
    all_cats=all_cats[(all_cats['FWHM_IMAGE']>3)&(all_cats['FLUX_RADIUS']>2.7)]
    
    # Match to observed objects (airy flight cats) in RA/Dec
    gal_match=htm.Matcher(16,ra=truthgals['ra'],dec=truthgals['dec'])
    all_ind,truth_ind, dist=gal_match.match(ra=all_cats['ALPHAWIN_J2000'],dec=all_cats['DELTAWIN_J2000'],maxmatch=1,radius=2.5E-4)
    
    print("Found %d real galaxies in catalog %s" %(len(all_ind),all_cats))

    master_gal_cat=hstack([all_cats[all_ind],truthgals[truth_ind]])
    
    return master_gal_cat

def isolate_real_gals(fullcat,starcat):
    try:
        star_matcher=htm.Matcher(16,ra=starcat['ALPHAWIN_J2000'],dec=starcat['DELTAWIN_J2000'])
    except:
        star_matcher=htm.Matcher(16,ra=starcat['ra'],dec=starcat['dec'])
    full_ind, star_ind, dist=star_matcher.match(ra=fullcat['ALPHAWIN_J2000'],dec=fullcat['DELTAWIN_J2000'],maxmatch=1,radius=2.5E-4)
    this_range=np.arange(len(fullcat)) 
    gals=np.setdiff1d(this_range,full_ind)
    gals_only = fullcat[gals]
    gals_only = gals_only[(gals_only['FWHM_IMAGE']>3) & (gals_only['FLUX_RADIUS']>2.7)] # clean out the junk
    
    return gals_only

###### 300 s exposures ###########

fullcats=glob.glob('/Users/jemcclea/Research/GalSim/examples/output/25.2_empirical/double_srcs_300_[0-4].ldac')
#fullcats=glob.glob('/Users/jemcclea/Research/GalSim/examples/output/23.5_empirical/mockSuperbit_nodilate_300_[0-4]_cat.ldac')
mock300=get_catalogs(fullcats)

#truth300cats=glob.glob('/Users/jemcclea/Research/GalSim/examples/output/23.5_empirical/truth_nodilate_300_[0-4].dat')
truth300cats=glob.glob('/Users/jemcclea/Research/GalSim/examples/output/25.2_empirical/truth_double_srcs_300_[0-4].dat')
truth300=get_catalogs(truth300cats)

stars300=get_stars(truth300,mock300)

gals300=get_galaxies(truth300,mock300) 

###### 150 s exposures ###########

full150=glob.glob('/Users/jemcclea/Research/GalSim/examples/output/nodilate_150_?.ldac')
mock150=get_catalogs(full150)

truth150cats=glob.glob('/Users/jemcclea/Research/GalSim/examples/output/truth_nodilate_150*dat')
truth150=get_catalogs(truth150cats)
full150_ind,stars150_ind=get_stars(truth150,mock150)
stars150=get_ellipticities(mock150,full150_ind,truth150,stars150_ind)
gal150_ind,truthgal150_ind=get_galaxies(truth150,mock150) 
gals150=get_ellipticities(mock150,gal150_ind,truth150,truthgal150_ind)


########## Real empirical catalogs, as well as mocks ###############

#real300cats=glob.glob('/Users/jemcclea/Research/SuperBIT/A2218/Clean/dwb_image*300*WCS_cat.ldac')
real300cats=glob.glob('/Users/jemcclea/Research/GalSim/examples/image_ifc_1*300*WCS.ldac')
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
real=Table.read('/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-real/coadd_cat.ldac',hdu=2)
#mock=Table.read('/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-empirical/23.5/mock_empirical_psf_coadd_cat_full.ldac',hdu=2)  
mock=Table.read('/Users/jemcclea/Research/GalSim/examples/output/25.2_empirical/double_srcs_coadd_cat.ldac',hdu=2)  

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
plt.loglog(gals300['flux'],gals300['stamp_sum'],'.b',alpha=0.5)
plt.xlim([1,5E5])
plt.xlabel('COSMOS flux'); plt.ylabel('stamp flux'); plt.legend()
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
plt.semilogx(realgals300['FLUX_AUTO'],realgals300['FWHM_IMAGE'],'.k',label='real gals') 
#plt.semilogx(gals300['flux'],gals300['fwhm'],'*',label='mock gals size scaled',alpha=0.5)
plt.semilogx(gals300['FLUX_AUTO'],gals300['FWHM_IMAGE'],'*',label='mock gals size',alpha=0.5)
plt.xlabel('Flux'); plt.ylabel('FWHM'); plt.legend()

plt.savefig('doublesrcs_300s_gals_sizemag.png')

plt.figure()
plt.hist(realgals300['FLUX_AUTO'],bins=80,alpha=0.5,range=[1,1E5],log=True, label='real 300s gals') 
#plt.hist(gals300['flux'],bins=80,alpha=0.5,range=[1,1E5],log=True,label='mock 300s gals')
plt.hist(gals300['FLUX_AUTO'],bins=80,alpha=0.5,range=[1,1E5],log=True,label='mock 300s gals')
plt.xlabel('FLUX_AUTO'); plt.ylabel('number'); plt.legend()

# Check the flux distribution for real & mock 300s full catalogs (not just galaxies)
plt.figure()
plt.hist(real300['FLUX_AUTO'][(real300['FWHM_IMAGE']>3)&(real300['FLUX_RADIUS']>2.7)],range=[1,2E5],bins=80,alpha=0.5,label='Real 300s FWHM>0.8',log=True)  
plt.hist(mock300['FLUX_AUTO'][(mock300['FWHM_IMAGE']>3)&(mock300['FLUX_RADIUS']>2.7)],range=[1,2E5],bins=80,alpha=0.5,label='Mock 300s FWHM>0.8',log=True)
plt.legend()
plt.xlabel('FLUX_AUTO'); plt.ylabel('Number')


plt.savefig('1.5x_source_count_flux_hist.png')

# Check the size-mag distributions for real & mock 300s full catalogs (not just galaxies)
plt.figure()
plt.semilogx(real300['FLUX_AUTO'][(real300['FWHM_IMAGE']>3)&(real300['FLUX_RADIUS']>2.7)],
                 real300['FWHM_IMAGE'][(real300['FWHM_IMAGE']>3)&(real300['FLUX_RADIUS']>2.7)],'.k',label='real 300s',alpha=0.5)
plt.semilogx(mock300['FLUX_AUTO'][(mock300['FWHM_IMAGE']>3)&(mock300['FLUX_RADIUS']>2.7)],mock300['FWHM_IMAGE'][(mock300['FWHM_IMAGE']>3)&(mock300['FLUX_RADIUS']>2.7)],'*b',label='mock 300s',alpha=0.5)
plt.xlabel('FLUX_AUTO'); plt.ylabel('FWHM_IMAGE'); plt.legend(); plt.ylim([-1,45]) 

#####################
### 150s checkplots
#####################

# size-mag and flux distributions for real & mock 150s galaxies
plt.figure()
plt.semilogx(realgals150['FLUX_AUTO'],realgals150['FWHM_IMAGE'],'.k',label='real gals') 
plt.semilogx(gals150['flux'],gals150['fwhm'],'*',label='mock gals',alpha=0.5)
plt.xlabel('Flux'); plt.ylabel('FWHM'); plt.legend()

# There's a lot of junk in the 150s catalogs, so use density
plt.figure()
plt.hist(real150['FLUX_AUTO'][(real150['FWHM_IMAGE']>4) & (real150['FLUX_RADIUS']>4)],
             alpha=0.5,bins=80,range=[1,2E5], label='Real 150s cat',log=True)
#plt.hist(mock150['MAG_AUTO'],bins=80,range=[1,2E5], label='Mock 150s cat',log=True,histtype='step')
plt.hist(mock150['FLUX_AUTO'],bins=80,range=[1,2E5], label='Mock 150s cat',log=True,alpha=0.5)

plt.legend()
plt.xlabel('FLUX_AUTO'); plt.ylabel('Number')


# Check the size-mag distributions for real & mock 150s full catalogs (not just galaxies)
plt.figure()
plt.semilogx(real150['MAG_AUTO'][(real150['FWHM_IMAGE']>4)&(real150['FLUX_RADIUS']>2.7)],
                 real150['FWHM_IMAGE'][(real150['FWHM_IMAGE']>4)&(real150['FLUX_RADIUS']>2.7)],'.k',label='real 150s',alpha=0.4)
plt.semilogx(mock150['MAG_AUTO'],mock150['FWHM_IMAGE']*1.5,'*b',label='mock 150s fwhm*1.5',alpha=0.4)
plt.xlabel('MAG_AUTO'); plt.ylabel('FWHM_IMAGE'); plt.legend(); plt.ylim([-1,60]) 


##########################
### stack image checkplots
##########################

# Do flux & mag histograms for real and mock stacks
plt.figure()
plt.hist(real['FLUX_AUTO'][(real['FWHM_IMAGE']>3) & (real['FLUX_RADIUS']>2.7)],bins=80,range=[1,1e5],alpha=0.5, label='Real stack cat FWHM > 0.8',log=True)
plt.hist(mock['FLUX_AUTO'][(mock['FWHM_IMAGE']>3)& (mock['FLUX_RADIUS']>2.7)],bins=80,alpha=0.5,range=[1,1e5],label='Mock stack cat FWHM > 0.8',log=True)
plt.legend()
plt.xlabel('FLUX_AUTO'); plt.ylabel('Number')

plt.savefig('dblsrc_stack_fluxhist.png')

plt.figure()
plt.hist(real['MAG_AUTO'][(real['FWHM_IMAGE']>3) & (real['FLUX_RADIUS']>2.7)],bins=70,range=[15,25], label='Real stack cat',alpha=0.5,log=True)
plt.hist(mock['MAG_AUTO'][(mock['FWHM_IMAGE']>3)& (mock['FLUX_RADIUS']>2.7)],bins=70,range=[15,25], label='Mock stack cat',alpha=0.5,log=True)
plt.legend()
plt.xlabel('MAG_AUTO'); plt.ylabel('Number')

# 
plt.figure()
plt.semilogx(real['FLUX_AUTO'][(real['FWHM_IMAGE']>3)&(real['FLUX_RADIUS']>2.7)],real['FWHM_IMAGE'][(real['FWHM_IMAGE']>3)&(real['FLUX_RADIUS']>2.7)],'.k',label='real stack',alpha=0.5)
plt.semilogx(mock['FLUX_AUTO'][(mock['FWHM_IMAGE']>3)& (mock['FLUX_RADIUS']>2.7)],mock['FWHM_IMAGE'][(mock['FWHM_IMAGE']>3)& (mock['FLUX_RADIUS']>2.7)],'*b',label='mock stack',alpha=0.5)
plt.xlabel('FLUX_AUTO'); plt.ylabel('FWHM_IMAGE'); plt.legend(); plt.ylim([-1,45]) 


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
########### Calculate depth of an observation -- DO NOT CHANGE ###############
##############################################################################################

# gals is the analysis object catalog
n,bins=np.histogram(gals['MAG_AUTO'],bins=100)
midpoints = np.array([(bins[i+1]+bins[i])*0.5 for i in range(len(bins)-1)])

wg=(midpoints<26.4) & (midpoints>19)

fit=np.polyfit(midpoints[wg],np.log(n[wg]),1)
num = fit[0]*midpoints+fit[1]

plt.hist(gals['MAG_AUTO'],histtype='step',bins=100,label='mock deep 3hr b',log=True)
plt.plot(midpoints,np.exp(num),'--k')

# OK, now to estimate 50% completeness
fraction=np.log(n)/num
enum=enumerate(fraction)  
l = list(enum)

# Here you have to pick your point in the resulting l array.
# In one instance, I used ind=80 for ~100% completeness,
# used ind=93 for 90% completeness, and np.mean(93,94) for 50% completeness

complete=midpoints[86]
complete90=midpoints[90]
complete50=np.mean([midpoints[97],midpoints[98]])


##############################################################################################
########### Have yourself a merry little filter profile plot  ###############
##############################################################################################

#LAMBO
lambo = np.arange(300,1090,1)

lum=Table.read('lum.csv')
u=Table.read('u.csv')
b=Table.read('b.csv')
g=Table.read('g.csv')
r=Table.read('r.csv')
i=Table.read('i.csv')

shape=Table.read('shape.dat')
u=Table.read('u.csv')
b=Table.read('b.csv')
g=Table.read('g.csv')
r=Table.read('r.csv')
i=Table.read('i.csv')


plt.figure()

plt.plot(lambo,u,color='m',label='u')
plt.plot(lambo,b,color='b',label='b')
plt.plot(lambo,g,color='g',label='g')
plt.plot(lambo,r,color='r',label='r') 
plt.plot(lambo,i,color='darkred',label='i')
plt.plot(lambo,lum,color='k',label='LUM')
plt.legend(loc='top left')
plt.xlabel('Wavelength (Angstroms)')
plt.ylabel('System Throughput')
plt.savefig('superbit_2019_filterprofiles.png')



## Alternatively...
cd '/Users/jemcclea/Research/SuperBIT/Telecon updates/BIT_2022'
u2=Table.read('u.csv',format='ascii')
b2=Table.read('b.csv',format='ascii')
g2=Table.read('g.csv',format='ascii')
r2=Table.read('r.csv',format='ascii')
i2=Table.read('i.csv',format='ascii')
shape=Table.read('shape.csv',format='ascii')                                                                                                                

plt.figure()
plt.plot(u2['col1'],u2['col2'],label='u',color='m')                                                                                                                       
plt.plot(b2['col1'],b2['col2'],label='b',color='b')                                                                                                                       
plt.plot(g2['col1'],g2['col2'],label='g',color='g')                                                                                                                       
plt.plot(r2['col1'],r2['col2'],label='r',color='r')                                                                                                                       
plt.plot(i2['col1'],i2['col2'],label='i',color='darkred')                                                                                                                 
plt.plot(shape['col1'],shape['col2'],label='shape',color='k')                                                                                                             

plt.legend()                                                                                                                                                              
plt.xlabel('Wavelength (Angstroms)')                                                                                                                                      
plt.ylabel('System Throughput (Normalized)')

plt.savefig('superbit_2022_filterprofiles.png')

## for when you want to compare old lum vs. new shape
vals=[float(lum[i][0]) for i in range(len(lum))]
vals=np.array(vals)
lum_norm=(vals/max(vals))*100
plt.plot(lambo,lum_norm,color='k',alpha=0.5,label='2019 LUM filter')
plt.plot(shape['col1'],shape['col2'],label='shape',color='k')
plt.legend()
plt.xlabel('Wavelength (Angstroms)')                                                                                                                                      
plt.ylabel('System Throughput (Normalized)')                                                                                                                              
plt.savefig('superbit_shape_vs_lum.png')

##############################################################################################
########### If you want to filter out stars from mock catalogs  ###############
##############################################################################################

fulln='/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-jitter/one_hour_obs/mock_coadd_cat_full.ldac'
full=Table.read(fulln,format='fits',hdu=2)

truthdir = '/Users/jemcclea/Research/SuperBIT/superbit-metacal/GalSim/cluster3-newpsf/round6'
truthcatn = 'truth_gaussJitter.002.dat'
#truthdir = '/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-jitter/one_hour_obs'
#truthcat = 'truth_superbit300004.dat'
truthfile=os.path.join(truthdir,truthcatn)

#truthfile = '/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-deep/truth_flight_jitter_only_oversampled_1x300.0006.dat'

truthcat = Table.read(truthfile,format='ascii')
stars=truthcat[truthcat['redshift']==0] 

star_matcher = htm.Matcher(16,ra=stars['ra'],dec=stars['dec'])
matches,starmatches,dist = star_matcher.match(ra=full['ALPHAWIN_J2000'],
                                                    dec=full['DELTAWIN_J2000'],radius=2E-4,maxmatch=1)

stars = full[matches]



# Save result to file, return filename
outname = fulln.replace('.ldac','.star')
full[matches].write(outname,format='fits',overwrite=True)


# Make cute plots, if desired:

from matplotlib import colors
from matplotlib.ticker import PercentFormatter

label = (r'$\langle fwhm^{\star} \rangle $ = %.4f'%np.median(stars['FWHM_WORLD']*3600))
plt.hist(stars['FWHM_WORLD']*3600, bins=30,range=[0.35,0.85],label=label)
plt.xlabel('stellar FWHM (arseconds)')


fig, axs = plt.subplots(1, 1, tight_layout=True)
# N is the count in each bin, bins is the lower-limit of the bin
N, bins, patches = axs.hist(stars['FWHM_WORLD']*3600, bins=30,range=[0.35,0.85],label=label)

# We'll color code by height, but you could use any scalar
fracs = N / N.max()

# we need to normalize the data to 0..1 for the full range of the colormap
norm = colors.Normalize(fracs.min(), fracs.max())
len
# Now, we'll loop through our objects and set the color of each accordingly
for thisfrac, thispatch in zip(fracs, patches):
    color = plt.cm.viridis(norm(thisfrac))
    thispatch.set_facecolor(color)
plt.xlabel('stellar FWHM (arseconds)') 
    
fig.savefig('1.5hr_stellar_fwhm.png')


##############################################################################################
########### Debug T/sigma plots  ###############
##############################################################################################

gals = Table.read('superbit_gaussStars_006_cat.ldac',hdu=2)
mcal = Table.read('/Users/jemcclea/Research/SuperBIT/shear_profiles/stars/GaussPSF')
r = np.sqrt((newtab['X_IMAGE']-3505)**2 + (newtab['Y_IMAGE']-2340)**2)
number,bins = np.histogram(r,bins=20,range=(5,3000))

sigma_sex = (gals['FWHM_WORLD']*3600)/2.355
sigma_mcal = np.sqrt(mcal['psf_T']/2)

diff = np.sqrt((sigma_sex - sigma_fitvd)**2) 
ratio = sigma_sex/sigma_fitvd

diffs = []
ratios = []
errs_diff = []
errs_ratio = []
midpoints_r = []
for i in range(len(bins)-1):
    annulus = (r>=bins[i]) & (r<bins[i+1])          
    midpoint_r = np.mean([bins[i],bins[i+1]])
    midpoints_r.append(midpoint_r)
    n = number[i]
    this_diff = diff[annulus][~np.isnan(diff[annulus])]
    this_ratio = ratio[annulus][~np.isnan(ratio[annulus])]
    
    diff_mean = np.mean(this_diff)
    ratio_mean = np.mean(this_ratio)

    diffs.append(diff_mean); ratios.append(ratio_mean)
    
    diff_err = np.std(this_diff)/np.sqrt(n)
    ratio_err = np.std(this_ratio)/np.sqrt(n)

    errs_diff.append(diff_err); errs_ratio.append(ratio_err)

midpoints_r = np.array(midpoints_r); diffs = np.array(diffs); ratios = np.array(ratios)
errs_diff = np.array(errs_diff); errs_ratio = np.array(errs_ratio)

plt.errorbar(midpoints_r*.206/60,diffs,yerr=diff_err,fmt='-o',capsize=5,label=r'diffs')
plt.errorbar(midpoints_r*.206/60,ratios,yerr=ratio_err,fmt='-o',capsize=5,label=r'ratios')



##############################################################################################
########### Making new fitvd catalogs, with X/Y info, for annular shear profiles #############
##############################################################################################

## In case I can't find it:
fitvd --seed 67857848 --config /Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/fitvd-superbit-gauss.yaml --output fitvd-opticsGaussJitter.fit cluster3_debug_2hr.meds

#gals=Table.read('mock_empirical_psf_coadd_cat.ldac',format='fits',hdu=2) #contains only analysis objects.
#full=Table.read('mock_empirical_psf_coadd_cat_full.ldac',format='fits',hdu=2)

gals=Table.read('mock_coadd_cat.ldac',format='fits',hdu=2) #contains only analysis objects.
full=Table.read('mock_coadd_cat_full.ldac',format='fits',hdu=2)

plt.plot(full['MAG_AUTO'],full['FWHM_IMAGE'],'.b',alpha=0.4,label='all objects') 
plt.plot(gals['MAG_AUTO'],gals['FWHM_IMAGE'],'.r',alpha=0.4,label='analysis objects')

#plt.plot(stars['MAG_AUTO'],stars['FWHM_IMAGE'],'*b',alpha=0.4,label='stars') 

# Quality cuts
#truthcat=Table.read('/Users/jemcclea/Research/SuperBIT/superbit-metacal/GalSim/output-gaussian/truth_gaussian300.0001.dat',format='ascii')
truthcat=Table.read('truth_gaussJitter_004.dat',format='ascii')
bg_gals = truthcat[truthcat['redshift']>0.45]
gal_matcher = htm.Matcher(16,ra=bg_gals['ra'],dec=bg_gals['dec'])
matches,bg_galmatches,dist = gal_matcher.match(ra=gals['ALPHAWIN_J2000'],
                                                    dec=gals['DELTAWIN_J2000'],radius=6E-4,maxmatch=1)
print(len(matches))

gals=gals[matches]
gals = gals[gals['FWHM_IMAGE']>2]

fitvd=Table.read('/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-stars/fitvd-stars.fit',format='fits',hdu=1)
#fitvd_success =  (fitvd['exp_T']>=1.15*fitvd['psf_T']) #&  (fitvd['psf_flux_s2n'] >5) #
fitvd_success =(fitvd['gap_flux']>0) & (fitvd['gauss_s2n'] >10)
print(len(fitvd[fitvd_success]))

fitvd=fitvd[fitvd_success]

fitvd_matcher = htm.Matcher(16,ra=fitvd['ra'],dec=fitvd['dec'])
gals_ind,fitvd_ind,dist=fitvd_matcher.match(ra=gals['ALPHAWIN_J2000'],dec=gals['DELTAWIN_J2000'],radius=6E-4,maxmatch=1)
print(len(gals_ind))

fitvd=fitvd[fitvd_ind]; gals=gals[gals_ind]

newtab=Table()
newtab.add_columns([fitvd['id'],fitvd['ra'],fitvd['dec'],fitvd['gauss_g'][:,0],fitvd['gauss_g'][:,1]],names=['id','ra','dec','g1','g2'])
newtab.add_columns([fitvd['gauss_T'],fitvd['gauss_flux']],names=['T','flux'])
newtab.add_columns([fitvd['psf_T']],names=['psf_T'])
newtab.add_columns([gals['X_IMAGE'],gals['Y_IMAGE']])

newtab.write('fitvd-cl3-gaussPSF.csv',format='csv',overwrite=True) 
#newtab.write('/Users/jemcclea/Research/SuperBIT/superbit-ngmix/scripts/output-empirical/25.2/fitvd-empirical-gauss.csv',format='csv',overwrite=True)

## Convert to fiatformat:
cmd='sdsscsv2fiat fitvd-cl3-gaussPSF.csv > fitvd-cl3-gaussPSF.fiat'
os.system(cmd)

## On command line 
annular -c"X_IMAGE Y_IMAGE g1 g2" -s 100 -e 1500 -n 5 fitvd-empirical-gauss.fiat 3371.5 4078.5 #3505 2340 #
                   
annular -c"X_IMAGE Y_IMAGE g1 g2" -s 50 -e 2500 -n 15 fitvd-cl3-gaussPSF 3505 2340  #>fitvd-cl3-gaussPSF_center1.annular

annular -c"X_IMAGE Y_IMAGE X2_IMAGE Y2_IMAGE XY_IMAGE" -s 50 -e 2200 -n 15 mock_coadd_bgGals_Sexcat.fiat 3505 2340 #> opticalGaussJitter_nodilate_sexmoments.annular
python ../annular_jmac.py fitvd-cluster2.fiat X_IMAGE Y_IMAGE g1 g2

#for cluster 2 3600 2361
3514 2440
3505 2340
3540 2420
3540 2392
