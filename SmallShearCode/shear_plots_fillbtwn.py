import numpy as np
from matplotlib import rc,rcParams
rc('font',**{'family':'serif'})
rc('text', usetex=True)
import matplotlib.pyplot as plt
from astropy.table import Table
import pdb

class ShearProfilePlotter(object):

    def __init__(self, cat_file, pix_scale=0.141):
        '''
        cat_file: str
            Filename for binned shear profile data.
            Can optionally include truth information as well
            in this table for plot comparison
        pix_scale: float
            Pixel scale (arcsec / pix)
        '''

        self.cat_file = cat_file
        self.pix_scale = pix_scale

        self._load_cats()

        return

    def _load_cats(self):

        if isinstance(self.cat_file,str):
            self.cat = Table.read(self.cat_file)
        else:
            self.cat = self.cat_file

        return

    def get_angular_radius(self, pix_radius, arcmin=True):
        angular_radius = pix_radius * self.pix_scale

        if arcmin is True:
            return angular_radius / 60.
        else:
            # in arcsec
            return angular_radius

    def plot_tan_profile(self, title=None, size=(10,7), label=r'annular',
                         rbounds=(5, 750), show=False, outfile=None,
                         nfw_label=None, smoothing=False, plot_truth=True,
                         xlim=None, ylim=None,fill_between=True, shear_cutoff=None):
        '''
        xlim/ylim: list of tuples
            A list of len 2 containing the xlim/ylim boundaries for both plots;
            e.g. ylim=[(-1,5), (-2,2)]
        '''

        cat = self.cat

        # in arcsec
        minrad = rbounds[0]
        maxrad = rbounds[1]

        cat.sort('midpoint_r') # get in descending order

        radius = self.get_angular_radius(cat['midpoint_r'], arcmin=True)

        gtan = cat['mean_gtan']
        gcross = cat['mean_gcross']
        gtan_err = cat['err_gtan']
        gcross_err = cat['err_gcross']

        if plot_truth is True:
            try:
                # see if truth info is present
                true_gtan = cat['mean_nfw_gtan']
                true_gcross = cat['mean_nfw_gcross']
                true_gtan_err = cat['err_nfw_gtan']
                true_gcross_err = cat['err_nfw_gcross']
                true_radius = radius

            except KeyError:
                print('WARNING: Truth info not present in shear profile table!')
                plot_truth = False

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

        axs[0].errorbar(radius, gtan, yerr=gtan_err, fmt='-o',
                        capsize=5, color='cornflowerblue', label=label)

        if fill_between == True:
            shear_hi = gtan+gtan_err
            shear_low =  gtan-gtan_err
            axs[0].fill_between(radius, y1=shear_hi, y2=shear_low, alpha=0.1,color='darkturquoise')

        # If truth info is present, plot it
        if plot_truth is True:
            if smoothing is True:
                true_gtan = np.convolve(true_gtan, np.ones(5)/5, mode='valid')
                true_radius = np.convolve(true_radius, np.ones(5)/5, mode='valid')

            true_label = 'Reference NFW (resample)'
            axs[0].plot(true_radius, true_gtan, '-r', label=true_label)

            # grab alpha statistics
            alpha = cat.meta['alpha']
            sig_alpha = cat.meta['sig_alpha']

            txt = str(r'$\hat{\alpha}=%.4f~\sigma_{\hat{\alpha}}=%.4f$' % (alpha, sig_alpha))
            ann = axs[0].annotate(
                txt, xy=[0.1,0.9], xycoords='axes fraction', fontsize=12,
                bbox=dict(facecolor='white', edgecolor='cornflowerblue',
                          alpha=0.8,boxstyle='round,pad=0.3')
                )

            if shear_cutoff is not None:
                sc_label = rf'$\alpha$ calc cutoff = {shear_cutoff:.3f}'
                axs[0].axhline(shear_cutoff, c='black', alpha=0.9, linestyle='dotted', label=sc_label)

        # reference line
        axs[0].axhline(y=0, c="black", alpha=0.4, linestyle='--')

        axs[0].set_ylabel(r'$g_{+}(\theta)$', fontsize=16)
        axs[0].tick_params(which='major', width=1.3, length=8)
        axs[0].tick_params(which='minor', width=0.8, length=4)
        axs[0].legend()

        axs[1].errorbar(radius, gcross, yerr=gcross_err, fmt='d',
                        capsize=5, color='cornflowerblue', label=label)
        axs[1].axhline(y=0, c="black", alpha=0.4, linestyle='--')
        axs[1].set_xlabel(r'$\theta$ (arcmin)', fontsize=16)
        axs[1].set_ylabel(r'$g_{\times}(\theta)$', fontsize=16)
        axs[1].tick_params(which='major', width=1.3, length=8)
        axs[1].tick_params(which='minor', width=0.8, length=4)
        axs[1].legend()

        if xlim is not None:
            for i in range(2):
                xl = xlim[i]
                axes[i].set_xlim(xl[0], xl[1])
        if ylim is not None:
            for i in range(2):
                yl = ylim[i]
                axes[i].set_ylim(yl[0], yl[1])

        if title is None:
            axs[0].set_title(title, fontsize=14)

        if outfile is not None:
            plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if show is True:
            plt.show()

        return
