### VIP: START WITH $ module load texlive/2018 first! (on CCV)

import numpy as np
from matplotlib import rc,rcParams
#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('font',**{'family':'serif'})
rc('text', usetex=True)

import matplotlib.pyplot as plt
from astropy.table import Table

plt.ion()


data=Table.read('/Users/jemcclea/Research/SuperBIT/shear_profiles/debug3/metacal-debug3-shear-MC.annular',format='ascii')
data.sort('col1') # get in descending order
radius=data['col1']#[:-1]
radius=radius*.206/60.

etan=data['col6']
ecross=data['col7']
shear1err=data['col8']
shear2err=data['col9']
"""
etan=data['col3']
ecross=data['col4']
shear1err=data['col5']
shear2err=data['col6']
"""



# So far, this is looking great!!! Now, let's remember how to make bin width error bars
minrad = 200*.206/60 #pixels --> arcmin
maxrad = 4000*.206/60 #pixels --> arcmin

upper_err=np.zeros_like(radius)
for e in range(len(radius)-1):
    this_err=(radius[e+1]-radius[e])*0.5
    upper_err[e]=this_err
upper_err[-1]=(maxrad-radius[-1])*0.5

lower_err=np.zeros_like(radius)
for e in (np.arange(len(radius)-1)+1):
    this_err=(radius[e]-radius[e-1])*0.5
    lower_err[e]=this_err

lower_err[0]=(radius[0]-minrad)*0.5

# And scene
#rad_err=np.vstack([lower_err[1:],upper_err[1:]])
rad_err=np.vstack([lower_err,upper_err])

rcParams['axes.linewidth'] = 1.3
rcParams['xtick.labelsize'] = 16
rcParams['ytick.labelsize'] = 16
rcParams['xtick.minor.visible'] = True
rcParams['xtick.minor.width'] = 1
rcParams['xtick.direction'] = 'inout'
rcParams['ytick.minor.visible'] = True
rcParams['ytick.minor.width'] = 1
rcParams['ytick.direction'] = 'out'


fig, axs = plt.subplots(2,1,figsize=(10,6),sharex=True)#,sharey=True)
fig.subplots_adjust(hspace=0.1)

axs[0].errorbar(radius,etan,yerr=shear1err,xerr=rad_err,fmt='o',capsize=5, label=r'Mcal ``$g_{MC}$" PsfScale=0.538')
axs[0].axhline(y=0,c="black",alpha=0.4,linestyle='--')
axs[0].set_ylabel(r'$g_{+}(\theta)$',fontsize=16)
axs[0].tick_params(which='major',width=1.3,length=8)
axs[0].tick_params(which='minor',width=0.8,length=4)
axs[0].set_title('Gaussian PSF (Debug) 5E15 NFW Shear Signal',fontsize=16)
axs[0].set_ylim(-0.05,0.4)

axs[1].errorbar(radius,ecross,xerr=rad_err,yerr=shear2err,fmt='d',capsize=5,alpha=0.5,label=r'Mcal ``$g_{MC}$" PsfScale=0.538')
axs[1].axhline(y=0,c="black",alpha=0.4,linestyle='--')
axs[1].set_xlabel(r'$\theta$ (arcmin)',fontsize=16)
axs[1].set_ylabel(r'$g_{\times}(\theta)$',fontsize=16)
axs[1].tick_params(which='major',width=1.3,length=8)
axs[1].tick_params(which='minor',width=0.8,length=4)
axs[1].set_ylim(-0.1,0.1)
axs[1].legend()



#####
##### Overplotting? Do it again!
#####


data2=Table.read('/Users/jemcclea/Research/SuperBIT/shear_profiles/debug3/metacal-debug3-shear-gmix.annular',format='ascii')
data2.sort('col1') # get in descending order
radius2=data2['col1']#[:-1]
radius2=radius2*.206/60.

etan2=data2['col6']#[:-1]
ecross2=data2['col7']#[:-1]
shear1err2=data2['col8']#[:-1]
shear2err2=data2['col9']#[:-1]

"""
etan2=data2['col3']
shear1err2=data2['col4']
ecross2=data2['col7']
shear2err2=data2['col9']
"""

# So far, this is looking great!!! Now, let's remember how to make bin width error bars
minrad2 = 200*.206/60 #pixels --> arcmin
maxrad2 = 4000*.206/60 #pixels --> arcmin

upper_err=np.zeros_like(radius2)
for e in range(len(radius2)-1):
    this_err=(radius2[e+1]-radius2[e])*0.5
    upper_err[e]=this_err
upper_err[-1]=(maxrad-radius2[-1])*0.5

lower_err=np.zeros_like(radius2)
for e in (np.arange(len(radius2)-1)+1):
    this_err=(radius2[e]-radius2[e-1])*0.5
    lower_err[e]=this_err

lower_err[0]=(radius2[0]-minrad)*0.5

# And scene
#rad_err=np.vstack([lower_err[1:],upper_err[1:]])
rad_err2=np.vstack([lower_err,upper_err])

#pre-PSF truthcat shape

axs[0].errorbar(radius2,etan2,yerr=shear1err2,xerr=rad_err2,fmt='s',capsize=5,label=r'Metacal ``$g_{gmix}$" psfscale = 0.538')
axs[1].errorbar(radius2,ecross2,xerr=rad_err2,yerr=shear2err2,fmt='s',capsize=5,alpha=0.5,label=r'Metacal ``$g_{gmix}$" psfscale = 0.538')

axs[1].legend()


nfw_plotted2=Table.read('/Users/jemcclea/Research/SuperBIT/shear_profiles/debug3/nfw_plotted_c4.txt',format='ascii') 
nfw_r2 = nfw_plotted2['col1'] * .206/60.
axs[0].plot(nfw_r2[3:],nfw_plotted2['col2'][3:],'-r',label='5E15 NFW c=4')
axs[0].set_xlim(0.4,13.75)

axs[0].legend()

fig.savefig('5e15NFW_metacalMC_shear.png')
