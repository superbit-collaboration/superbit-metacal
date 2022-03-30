import numpy as np
from matplotlib import rc,rcParams
import matplotlib.pyplot as plt
from astropy.table import Table
import pudb

def compute_alpha(nfw, radius, gtan, variance):
    '''
    '''

    nfwr = nfw[0]
    nfw_shear = nfw[1]
    C = np.diag(variance**2)
    D = gtan

    # build theory array to match data points with a kludge
    # radius defines the length of our dataset
    T = []
    for rstep in radius:
        T.append(np.interp(rstep, nfwr, nfw_shear))
    T = np.array(T)

    numer = T.T.dot(np.linalg.inv(C)).dot(D)
    denom = T.T.dot(np.linalg.inv(C)).dot(T)

    Ahat = numer / denom
    sigma_A = 1./np.sqrt((T.T.dot(np.linalg.inv(C)).dot(T)))

    return Ahat, sigma_A

class ShearProfilePlotter(object):

    def __init__(self, cat_file, truth_file, pix_scale=0.144):
        '''
        cat_file: str
            Filename for binned shear profile data. Usually in
            the form of `{run_name}_transformed_shear_tab.fits`
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

    def get_angular_radius(self, pix_radius, arcmin=True):
        angular_radius = pix_radius * self.pix_scale

        if arcmin is True:
            return angular_radius * 60.
        else:
            # in arcsec
            return angular_radius

    def plot_tan_profile(self, title=None, size=(10,7), label='annular',
                         rbounds=(5, 750), show=False, outfile=None):
        '''
        rbounds: tuple
            A tuple of the form (rmin, rmax) in arcsec
        '''

        rc('font', **{'family':'serif'})
        rc('text', usetex=True)
        plt.ion()

        cat = self.cat
        minrad = rbounds[0]
        maxrad = rbounds[1]

        cat.sort('midpoint_r') # get in descending order

        radius = self.get_angular_radius(cat['midpoint_r'], arcmin=True)

        gtan = cat['gtan_mean']
        gcross = cat['gcross_mean']
        gtan_err = cat['gtan_err']
        gcross_err = cat['gcross_err']

        # Compute errors
        upper_err = np.zeros_like(radius)
        for e in range(len(radius)-1):
            this_err = (radius[e+1]-radius[e])*0.5
            upper_err[e] = this_err
        upper_err[-1] = (maxrad-radius[-1])*0.5

        lower_err = np.zeros_like(radius)
        for e in (np.arange(len(radius)-1)+1):
            this_err = (radius[e]-radius[e-1])*0.5
            lower_err[e] = this_err

        lower_err[0] = (radius[0]-minrad)*0.5

        rad_err = np.vstack([lower_err,upper_err])

        rcParams['axes.linewidth'] = 1.3
        rcParams['xtick.labelsize'] = 16
        rcParams['ytick.labelsize'] = 16
        rcParams['xtick.minor.visible'] = True
        rcParams['xtick.minor.width'] = 1
        rcParams['xtick.direction'] = 'inout'
        rcParams['ytick.minor.visible'] = True
        rcParams['ytick.minor.width'] = 1
        rcParams['ytick.direction'] = 'out'

        fig, axs = plt.subplots(2, 1, figsize=size, sharex=True)
        fig.subplots_adjust(hspace=0.1)

        axs[0].errorbar(radius, gtan, xerr=rad_err, yerr=gtan_err, fmt='-o',
                        capsize=5, color='cornflowerblue', label=label)
        axs[0].axhline(y=0, c="black", alpha=0.4, linestyle='--')
        axs[0].set_ylabel(r'$g_{+}(\theta)$', fontsize=16)
        axs[0].tick_params(which='major', width=1.3, length=8)
        axs[0].tick_params(which='minor', width=0.8, length=4)
        axs[0].set_ylim(-0.05, 0.60)

        axs[1].errorbar(radius, gcross, xerr=rad_err, yerr=gcross_err, fmt='d',
                        capsize=5, color='cornflowerblue', alpha=0.5, label=label)
        axs[1].axhline(y=0, c="black", alpha=0.4, linestyle='--')
        axs[1].set_xlabel(r'$\theta$ (arcmin)', fontsize=16)
        axs[1].set_ylabel(r'$g_{\times}(\theta)$', fontsize=16)
        axs[1].tick_params(which='major', width=1.3, length=8)
        axs[1].tick_params(which='minor', width=0.8, length=4)
        axs[1].set_ylim(-0.1, 0.1)
        axs[1].legend()

        if title is None:
            axs[0].set_title(title, fontsize=14)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        return
