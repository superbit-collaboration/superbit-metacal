import numpy as np
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
from astropy.table import Table
import pdb

class ShearProfilePlotter(object):

    def __init__(self, cat_file, truth_file, pix_scale=0.144):
        '''
        cat_file: str
            Filename for...
        truth_file: str
            Filename for nfw truth shear profile
        pix_scale: float
            Pixel scale (arcsec / pix)
        '''

        self.cat_file = cat_file
        self.truth_file = truth_file
        self.pix_scale = pix_scale

        self._load_cats()

        return

    def _load_cats(self):
        self.cat = Table.read(self.cat_file)
        self.truth = Table.read(self.truth_file)

        return

    def get_angular_radius(pix_radius, arcmin=True):
        angular_radius = pix_radius * self.pix_scale

        if arcmin is True:
            return angular_radius * 60.
        else:
            # in arcsec
            return angular_radius

    def plot_tan_profile(self, title=None, size=(10,7),
                         label='annular', show=False, outfile=None):
        rc('font', **{'family':'serif'})
        rc('text', usetex=True)
        plt.ion()


        radius = get_angular_radius(self.cat['r'], arcmin=True)
        gtan = self.cat['gtan']
        gcross = self.cat['gcross']
        gtan_err = self.cat['gtan_err']
        gcross_err = self.cat['gcross_err']

        try:
            data.sort('col1') # get in descending order
            ## uncomment below if it's needed
            #data.remove_rows([0])
            radius=data['col1']
            radius=radius*pixscale/60.
        except ValueError:
            data.sort('r') # get in descending order
            ## uncomment below if it's needed
            #data.remove_rows([0])
            radius=data['r']
            radius=radius*pixscale/60.

        try:
            etan=data['gtan']
            ecross=data['gcross']
            shear1err=data['err_gtan']
            shear2err=data['err_gcross']

        except KeyError:

            etan=data['col3']
            ecross=data['col4']
            shear1err=data['col5']
            shear2err=data['col6']

        # So far, this is looking great!!! Now, let's remember how to make bin width error bars
        minrad = minrad*pixscale/60 #pixels --> arcmin
        maxrad = maxrad*pixscale/60 #pixels --> arcmin

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

        fig, axs = plt.subplots(2, 1, figsize=size,sharex=True)#,sharey=True)
        fig.subplots_adjust(hspace=0.1)

        axs[0].errorbar(radius, etan, yerr=shear1err, xerr=rad_err, fmt='-o',
                        capsize=5, color='cornflowerblue', label=label)
        axs[0].axhline(y=0,c="black",alpha=0.4,linestyle='--')
        axs[0].set_ylabel(r'$g_{+}(\theta)$',fontsize=16)
        axs[0].tick_params(which='major',width=1.3,length=8)
        axs[0].tick_params(which='minor',width=0.8,length=4)
        axs[0].set_title(title,fontsize=14)
        axs[0].set_ylim(-0.05,0.60)

        axs[1].errorbar(radius,ecross,xerr=rad_err,yerr=shear2err,fmt='d',capsize=5,color='cornflowerblue',alpha=0.5,label=label)
        axs[1].axhline(y=0,c="black",alpha=0.4,linestyle='--')
        axs[1].set_xlabel(r'$\theta$ (arcmin)',fontsize=16)
        axs[1].set_ylabel(r'$g_{\times}(\theta)$',fontsize=16)
        axs[1].tick_params(which='major',width=1.3,length=8)
        axs[1].tick_params(which='minor',width=0.8,length=4)
        axs[1].set_ylim(-0.1, 0.1)
        axs[1].legend()

        if title is None:
            title = r'S2N\_R$>$5 T$>$ 1.2*TPsf $0.02<T<10$ 1E-2 covcut'

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        return

def covar_calculations(nfw, radius, etan, variance):

    # set some definitions; this will obviously throw horrible errors if it receives the wrong object

    nfwr = nfw[0]
    nfw_shear = nfw[1]
    C = np.diag(variance**2)
    D = etan

    # build theory array to match data points with a kludge
    # radius defines the length of our dataset
    T = []
    for rstep in radius:
        T.append(np.interp(rstep, nfwr, nfw_shear))
    T = np.array(T)

    # Ok, I think we are ready
    Ahat = T.T.dot(np.linalg.inv(C)).dot(D)/ (T.T.dot(np.linalg.inv(C)).dot(T))
    sigma_A = 1./np.sqrt((T.T.dot(np.linalg.inv(C)).dot(T)))

    return Ahat, sigma_A
