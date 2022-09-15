from astropy.table import vstack, hstack, Table
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from astropy.io import fits
import glob
from esutil import htm
import sys
import os
import astropy
import collections
from argparse import ArgumentParser
from statsmodels.stats.weightstats import DescrStatsW

import superbit_lensing.utils as utils

import ipdb

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('-shear_cats', type=str, default=None,
                        help = 'Tables to read in: xxx_transformed_shear_tab.fits')
    parser.add_argument('-nfw_file', type=str, default=None,
                        help='Reference NFW shear catalog')
    parser.add_argument('-shear_cut', type=float, default=None,
                        help='Max tangential shear to define scale cuts')
    parser.add_argument('-outfile', type=str, default=None,
                        help = 'Name of ouput stacked catalog')
    parser.add_argument('-start_rad', type=float, default=100,
                        help='Starting radius value (in pixels)')
    parser.add_argument('-end_rad', type=float, default=5200,
                        help='Ending radius value (in pixels)')
    parser.add_argument('-nbins', type=int, default=18,
                        help='Number of radial bins')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Overwrite output mcal file')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Show plots')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Make verbose')

    return parser.parse_args()

class CatalogStacker():

    def __init__(self, cat_list=None):

        self.cat_list = cat_list
        self.alpha_list = []
        self.stacked_cat = None
        self.mean_a = None
        self.std_a = None
        self.avg_nobj = None
        self.num_alphas = None

    def _get_avg_nobj(self):
        '''
        Returns the average length of catalogs in cat_list.
        If cat_list is galaxy catalogs,  it's an average source density
        if cat_list is shear profile catalogs, it's the average number of radial bins
        '''

        len_stacked = len(self.stacked_cat)
        len_catlist = len(self.cat_list)
        avg_cat_len = len_stacked/len_catlist

        self.avg_nobj = avg_cat_len

        return

    def _get_alpha_stats(self):

        num_alphas  = len(self.alpha_list)
        mean_alpha = np.mean(self.alpha_list)
        std_alpha = np.std(self.alpha_list)/np.sqrt(num_alphas)

        self.mean_a = mean_alpha
        self.std_a = std_alpha
        self.num_alphas = num_alphas

        return

    def get_catalogs(self, catnames=None):
        '''
        Either stack the supplied catnames, or
        stack the catalogs already in self
        '''

        if catnames == None:
            catnames = self.cat_list

        assert isinstance(catnames, list), f"catnames should be a list, got {type(catnames)} instead"

        holding = {}

        for i in np.arange(len(catnames)):
            tab=Table.read(catnames[i],format='fits')
            holding["tab{0}".format(i)] = tab

            # Skip the alpha list if it doesn't exist
            try:
                self.alpha_list.append(tab.meta['ALPHA'])
            except KeyError:
                pass

        stacked_catalog = vstack([holding[val] for val in holding.keys()], metadata_conflicts='silent')
        stacked_catalog.meta = collections.OrderedDict()

        self.stacked_cat = stacked_catalog

        return

    def _get_cat_stats(self):

        self._get_avg_nobj()

        if (len(self.alpha_list) > 1):
            self._get_alpha_stats()

        return

    def run(self):

        # Concatenate all the catalogs
        self.get_catalogs()

        # Get summary statistics
        self._get_cat_stats()

        return

def compute_profile(catalog, minrad, maxrad, nbins, nfw_file):
    '''
    Compute the weighted average tangential and cross shears
    of the suppled catalog in radial bins.

    Format is assumed to be same as the [run_name]_transformed_shear_tab.fits
    output by the shear_profile module

    If an NFW shear catalog  is provided, compute radial shear profile
    for the NFW profile as well
    '''

    bins = np.linspace(minrad, maxrad, nbins)

    counts, bins = np.histogram(catalog['r'], bins=bins)

    N = len(bins) - 1
    midpoint_r = np.zeros(N)
    gtan_mean = np.zeros(N)
    gtan_err = np.zeros(N)
    gcross_mean = np.zeros(N)
    gcross_err = np.zeros(N)


    i = 0
    for b1, b2 in zip(bins[:-1], bins[1:]):

        annulus = (catalog['r'] >= b1) & (catalog['r'] < b2)
        n = counts[i]
        midpoint_r[i] = np.mean([b1, b2])

        weighted_gtan_stats = DescrStatsW(catalog['gcross'][annulus],
            weights=catalog['weight'][annulus], ddof=0)

        weighted_gcross_stats = DescrStatsW(catalog['gtan'][annulus],
            weights=catalog['weight'][annulus], ddof=0)

        gtan_mean[i] = weighted_gtan_stats.mean
        gcross_mean[i] = weighted_gcross_stats.mean

        gtan_err[i] = weighted_gtan_stats.std / np.sqrt(n)
        gcross_err[i] =  weighted_gcross_stats.std / np.sqrt(n)

        i += 1

    # Create table to store shear profile

    table = Table()
    table.add_columns(
        [counts, midpoint_r, gtan_mean, gcross_mean, gtan_err, gcross_err],
        names=['counts', 'midpoint_r', 'mean_gtan', 'mean_gcross', 'err_gtan', 'err_gcross']
        )

    # Repeat calculation if an nfw table is supplied
    if nfw_file is not None:

        nfw_tab = Table.read(nfw_file)

        nfw_mid_r = np.zeros(N)
        nfw_gtan_mean = np.zeros(N)
        nfw_gtan_err = np.zeros(N)
        nfw_gcross_mean = np.zeros(N)
        nfw_gcross_err = np.zeros(N)

        i = 0
        for b1, b2 in zip(bins[:-1], bins[1:]):
            annulus = (nfw_tab['r'] >= b1) & (nfw_tab['r'] < b2)
            n = counts[i]
            nfw_mid_r[i] = np.mean([b1, b2])
            nfw_gtan_mean[i] = np.mean(nfw_tab['gtan'][annulus])
            nfw_gcross_mean[i] = np.mean(nfw_tab['gcross'][annulus])
            nfw_gtan_err[i] = np.std(nfw_tab['gtan'][annulus]) / np.sqrt(n)
            nfw_gcross_err[i] = np.std(nfw_tab['gcross'][annulus]) / np.sqrt(n)

            i += 1

        # NFW shear profile should have same number of elements as galaxy shear profile
        assert(len(nfw_mid_r) == len(midpoint_r))

        # Add NFW profile to output table
        table.add_columns(
            [nfw_gtan_mean, nfw_gcross_mean, nfw_gtan_err, nfw_gcross_err],
            names=['mean_nfw_gtan', 'mean_nfw_gcross', 'err_nfw_gtan', 'err_nfw_gcross'],
            )

    return table

def shear_curve(r, a, b, c):
    '''
    A simple model for the tangential shear profile used to determine
    best estimate for where the profile crosses the shear cut
    '''

    # ratio = r / rs

    # return rho0 / ( ratio * (1 + ratio)**2 )

    # return a + b*(c**r)
    # return a + b*r + c*(r)**2
    return a * np.exp(-b * r) + c

def plot_curve_fit(pars, r, gtan_mean, err_gtan, mean_nfw_gtan, shear_cut_flag,
                   shear_cut=None, show=False, outfile=None, s=(9,6)):
    plt.errorbar(r, gtan_mean, err_gtan, c='tab:blue', label='Meas <gtan>')
    plt.plot(r, mean_nfw_gtan, c='tab:red', label='True <nfw_gtan>')
    plt.plot(r, shear_curve(r, *pars), ls=':', lw=2, c='k', label='Curve fit')

    # z = np.polyfit(r, gtan_mean, 3)
    # f = np.poly1d(z)
    # plt.plot(r, f(r), ls='--', label='Curve fit')

    if shear_cut is not None:
        rmin = np.min(r[~shear_cut_flag])
        plt.axvspan(0, rmin, facecolor='k', alpha=0.1)

    plt.axhline(0, lw=2, ls='--', c='k')
    plt.legend()

    plt.xlabel(r'$R$ (pixels)')
    plt.ylabel(r'<g_tan>')
    plt.title('Curve fit to mean shear profile')

    plt.gcf().set_size_inches(s)

    if outfile is not None:
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

    if show is True:
        plt.show()
    else:
        plt.close()

    return

def main(args):

    shearcat_names = args.shear_cats
    nfw_file = args.nfw_file
    shear_cut = args.shear_cut
    outfile = args.outfile
    minrad = args.start_rad
    maxrad = args.end_rad
    nbins = args.nbins
    show = args.show
    overwrite = args.overwrite
    vb = args.vb

    if shearcat_names is None:
        shearcat_names = 'r*/*_transformed_shear_tab.fits'
    if outfile is None:
        outfile = 'stacked_transformed_shear_tables.fits'

    if shear_cut is not None:
        if shear_cut <= 0:
            raise ValueError('shear_cut must be positive')

    outdir = os.path.dirname(outfile)
    logfile = 'mean_shear_profile.log'
    log = utils.setup_logger(logfile, logdir=outdir)
    logprint = utils.LogPrint(log, vb)

    shearcat_list = glob.glob(shearcat_names)


    # Get source density, shear cats and also average alpha
    all_shears = CatalogStacker(shearcat_list)
    all_shears.run()

    avg_n_sources = np.ceil(all_shears.avg_nobj)
    logprint('')
    logprint(f'Avg number of galaxies in catalog is {avg_n_sources}')
    logprint('')

    stacked_shear = all_shears.stacked_cat

    stacked_shear.sort('r')
    stacked_shear.write(outfile, format='fits', overwrite=overwrite)


    # Calculate mean shear profile, including the
    # NFW shear profile if such a catalog is provided

    shear_profile = compute_profile(catalog=stacked_shear, minrad=minrad,
                    maxrad=maxrad, nbins=nbins, nfw_file=nfw_file)



    #---------------------------------------------------------------------------
    # Now fit a curve to the mean profile to determine where to apply the
    # shear cut
    # pars, pars_cov = curve_fit(shear_curve, midpoint_r, gtan_mean, maxfev=2000)
    pars, pars_cov = curve_fit(
        shear_curve, shear_profile['midpoint_r'], shear_profile['mean_gtan'],
        sigma=shear_profile['err_gtan'], maxfev=20000,
        p0=(.1, .0001, .1)
        )
    logprint(f'pars: {pars}')

    if shear_cut is not None:
        shear_cut_flag = np.abs(shear_curve(shear_profile['midpoint_r'], *pars)) > shear_cut
    else:
        shear_cut_flag = np.zeros(len(shear_profile['midpoint_r']), dtype=bool)

    plt_outfile = os.path.join(outdir, 'mean_shear_profile_curve_fit.png')
    plot_curve_fit(
        pars, shear_profile['midpoint_r'], shear_profile['mean_gtan'],
        shear_profile['err_gtan'], shear_profile['mean_nfw_gtan'],
        shear_cut_flag, shear_cut=shear_cut, show=show, outfile=plt_outfile
        )

    #---------------------------------------------------------------------------
    # save cols & metadata

    shear_profile.add_columns(
        [shear_cut_flag], names=['shear_cut_flag']
        )

    if all_shears.mean_a is not None:
        logprint(f'mean alpha = {all_shears.mean_a:.5f} +/- ' +\
                 f'{all_shears.std_a/np.sqrt(all_shears.num_alphas):.4f}')
        logprint(f'std alpha = {all_shears.std_a:.5f}')
    else:
        print('No alphas found in supplied table')

    shear_profile.meta['avg_alpha'] = all_shears.mean_a
    shear_profile.meta['std_alpha'] = all_shears.std_a
    shear_profile.meta['num_alphas'] = all_shears.num_alphas
    shear_profile.meta['mean_n_gals'] = avg_n_sources

    mean_shear_name = outfile.replace('stacked','mean')
    logprint(f'Writing out mean shear profile catalog to {mean_shear_name}')
    shear_profile.write(mean_shear_name, format='fits', overwrite=overwrite)

    return

if __name__ == "__main__":
    args = parse_args()
    main(args)
