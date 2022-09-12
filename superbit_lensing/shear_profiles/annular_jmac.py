import sys
import getopt
import re
import os
import math
import numpy as np
from argparse import ArgumentParser
from numpy import r_, c_
from numpy import linalg as la
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from esutil import htm
from statsmodels.stats.weightstats import DescrStatsW


from shear_plots import ShearProfilePlotter

import ipdb

parser = ArgumentParser()

# NOTE: This remains for legacy calls. The main way to run the analysis is now with
#       the Annular class directly
parser.add_argument('annular_file', type=str,
                    help='Annular catalog filename')
parser.add_argument('outfile', type=str,
                    help='Output filename for tangential profile measurement')
parser.add_argument('g1_col', type=str,
                    help='Name of g1 column in annular file')
parser.add_argument('g2_col', type=str,
                    help='Name of g2 column in annular file')
parser.add_argument('start_rad', type=float,
                    help='Starting radius value (in pixels)')
parser.add_argument('end_rad', type=float,
                    help='Ending radius value (in pixels)')
parser.add_argument('nbins', type=int,
                    help='Number of radial bins')
parser.add_argument('-run_name', type=str, default=None,
                    help='Name of simulation run')
parser.add_argument('-outdir', type=str, default=None,
                    help='Output directory')
parser.add_argument('-truth_file', type=str, default=None,
                    help='Truth file containing redshift information')
parser.add_argument('-nfw_file', type=str, default=None,
                    help='Reference NFW shear catalog')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Set to overwrite output files')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Turn on for verbose prints')

def compute_shear_bias(profile_tab):
    '''
    Function to compute the max. likelihood estimator for the bias of a shear profile
    relative to the input NFW profile and the uncertainty on the bias.

    Saves the shear bias estimator ("alpha") and the uncertainty on the bias ("asigma")
    within the meta of the input profile_tab

    The following columns are assumed present in profile_tab:

    :gtan:      tangential shear of data
    :err_gtan:  RMS uncertainty for tangential shear data
    :nfw_gtan:  reference (NFW) tangential shear

    '''

    assert isinstance(profile_tab,Table)

    try:
        T = profile_tab['mean_nfw_gtan']
        D = profile_tab['mean_gtan']
        errbar = profile_tab['err_gtan']

    except KeyError as kerr:
        print('Shear bias calculation:')
        print('required columns not found; check input names?')
        raise kerr

    # C = covariance, alpha = shear bias maximum likelihood estimator
    C = np.diag(errbar**2)
    numer = T.T.dot(np.linalg.inv(C)).dot(D)
    denom = T.T.dot(np.linalg.inv(C)).dot(T)
    alpha = numer/denom

    # sigalpha: Cramer-Rao bound uncertainty on Ahat
    sig_alpha = 1. / np.sqrt((T.T.dot(np.linalg.inv(C)).dot(T)))

    print('# ')
    print(f'# shear bias is {alpha:.4f} +/- {sig_alpha:.3f}')
    print('# ')

    # add this information to profile_tab metadata
    profile_tab.meta.update({'alpha': alpha, 'sig_alpha': sig_alpha})

    return

class ShearCalc():

    def __init__(self, inputs=None):
        '''
        :inputs:    dictionary with x, y, g1, g2 columns
        :g1/2:     reduced-shear components
        :gcross:   cross-shear (B-mode)
        :gtan:     tangential shear (E-mode)
        :x/y:      x and y positions
        :r:        distance of galaxy to NFW center

        Ideally we would find a way to have these attributes inherited by Annular
        '''
        self.x = None
        self.y = None
        self.g1 = None
        self.g2 = None
        self.gcross = None
        self.gtan = None
        self.r = None

        if inputs is not None:
            self.x = inputs['x']
            self.y = inputs['y']
            self.g1 = inputs['g1']
            self.g2 = inputs['g2']

        return

    def get_r_gtan(self, xc, yc, apply_cut=True):
        '''
        Calculate distance from reference point located at (xc, yc) and rotate
        g1, g2 of galaxies into gtan, gcross

        xc: float
            x coordinate of reference point for shear calc
        yc: float
            y coordinate of reference point for shear calc
        apply_cut: bool
            Set to True to remove unphysical entries. This should be False
            if you want to preserve indexing & length, such as when using
            get_shear_cut()
        '''

        g = np.sqrt(self.g1**2 + self.g2**2)
        std_g = np.std(g)

        if apply_cut is True:
            wg = (g >= 0.)
            nbad = len(wg[wg < 0])
            print(f'## {nbad} of {len(g)} galaxies removed due to |g| < 0')
        else:
            # do nothing
            wg = np.ones(len(g), dtype=bool)

        self.g1 = self.g1[wg]
        self.g2 = self.g2[wg]
        self.x = self.x[wg]
        self.y = self.y[wg]

        self.r = np.sqrt(((self.x-xc)**2.) + ((self.y-yc)**2.))
        phi = np.arctan2((self.y-yc), (self.x-xc))

        print(f'## Mean |g|: {np.mean(g):.3f} sigma_|g|: {np.std(g):.3f}')

        self.gtan= -1.*(self.g1*np.cos(2.*phi) + self.g2*np.sin(2.*phi))
        # note that annular.c has opposite sign convention
        self.gcross = self.g1*np.sin(2.*phi) - self.g2*np.cos(2.*phi)

        return

class Annular(object):

    def __init__(self, cat_info, annular_info, nfw_info=None, run_name=None, vb=False):

        """
        :mcal_selected:   table that will be read in
        :outcat:   name of output shear profile catalog
        :truth_file: name of truth file with redshift info
        :g1/2:     reduced-shear components
        :gcross:   cross-shear (B-mode)
        :gtan:     tangential shear (E-mode)
        :x/y:      x and y positions
        :r:        distance of galaxy to NFW center
        :nbins:    number of radial bins for averaging (default = 5)
        :start/endrad: region to consider
        """

        self.cat_info = cat_info
        self.annular_info = annular_info
        self.nfw_info = nfw_info
        self.run_name = run_name
        self.n_truth_gals = None
        self.vb = vb
        self.g1 = None
        self.g2 = None
        self.gcross = None
        self.gtan = None
        self.weight = None
        self.ra = None
        self.dec = None
        self.x = None
        self.y = None
        self.r = None

        # may want to exclude radial bins whose mean gtan shear is
        # sufficiently large
        self.shear_cut = annular_info['shear_cut']
        self.shear_cut_cat = annular_info['shear_cut_cat']

        return

    def open_table(self, cat_info):

        annular_file = cat_info['mcal_selected']
        print(f'loading annular file {annular_file}')

        try:
            tab = Table.read(annular_file, format='fits')
        except:
            tab = Table.read(annular_file, format='ascii')
        try:
            self.x = tab[self.annular_info['xy_args'][0]]
            self.y = tab[self.annular_info['xy_args'][1]]
            self.g1 = tab[self.annular_info['shear_args'][0]]
            self.g2 = tab[self.annular_info['shear_args'][1]]
            self.ra = tab['ra']
            self.dec = tab['dec']
            self.weight = tab['weight']

        except Exception as e:
            print('Could not load xy/g1g2/radec columns; check supplied column names?')
            print(f'annular_info = {self.annular_info}')
            raise e

        return

    def transform_shears(self, outdir, overwrite=False):
        '''
        Create instance of ShearCalc class
        Compute radius from NFW center, gtan, and gcross
        '''

        shear_inputs = {
            'x': self.x,
            'y': self.y,
            'g1': self.g1,
            'g2': self.g2
            }

        xc = self.annular_info['coadd_center'][0]
        yc = self.annular_info['coadd_center'][1]

        shears = ShearCalc(inputs = shear_inputs)
        shears.get_r_gtan(xc=xc, yc=yc)

        # Now populate Annular object attributes with these columns
        x = shears.x ; y = shears.y
        self.r = shears.r
        self.gtan = shears.gtan
        self.gcross = shears.gcross

        newtab = Table()
        newtab.add_columns(
            [x, y, self.r, self.gtan, self.gcross, self.weight],
            names=['x', 'y', 'r', 'gcross', 'gtan', 'weight']
            )

        run_name = self.run_name
        outfile = os.path.join(
            outdir, f'{run_name}_transformed_shear_tab.fits'
            )
        newtab.write(outfile, format='fits', overwrite=overwrite)

        return

    def redshift_select(self):
        '''
        Select background galaxies from larger transformed shear catalog:
            - Load in truth file
            - Select background galaxies behind galaxy cluster
            - Match in RA/Dec to transformed shear catalog
            - Filter self.r, self.gtan, self.gcross to be background-only
            - Also store the number of galaxies injected into simulation
        '''

        truth_file = self.cat_info['truth_file']

        try:
            truth = Table.read(truth_file, format='fits')
            if self.vb is True:
                print(f'Read in truth file {truth_file}')

        except FileNotFoundError as fnf_err:
            print(f'truth catalog {truth_file} not found, check name/type?')
            raise fnf_err

        truth_gals = truth[truth['obj_class'] == 'gal']
        self.n_truth_gals = len(truth_gals)

        cluster_gals = truth[truth['obj_class']=='cluster_gal']
        cluster_redshift = np.mean(cluster_gals['redshift'])

        truth_bg_gals = truth[truth['redshift'] > cluster_redshift]

        truth_bg_matcher = htm.Matcher(16,
                                        ra = truth_bg_gals['ra'],
                                        dec = truth_bg_gals['dec']
                                        )

        ann_file_ind, truth_bg_ind, dist = truth_bg_matcher.match(
                                            ra = self.ra,
                                            dec = self.dec,
                                            maxmatch = 1,
                                            radius = 1./3600.
                                            )

        print(f'# {len(dist)} of {len(self.ra)} objects matched to ' +\
              'truth background galaxies')

        self.gtan = self.gtan[ann_file_ind]
        self.gcross = self.gcross[ann_file_ind]
        self.r = self.r[ann_file_ind]
        self.weight = self.weight[ann_file_ind]
        
        gal_redshifts = truth_bg_gals[truth_bg_ind]['redshift']

        return gal_redshifts

    def process_nfw(self, gal_redshifts, Nresample, outdir='.',
                    overwrite=False):
        '''
        Subsample theoretical galaxy redshift distribution to match redshift
        distribution of detected galaxy catalog using a sort of MC rejection
        sampling algorithm

        Also compute g_tan, g_cross

        Nresample: int
            Number of times to resample NFW profile redshifts
        '''

        if self.nfw_info is not None:

            # Resample so redshift distribution matches input galaxies
            nfw_tab = self._nfw_resample_redshift(
                gal_redshifts, Nresample, outdir=outdir, overwrite=overwrite,
                )

            # Calculate tangential and cross shears for nfw
            self._nfw_transform_shear(nfw_tab)

            nfw_tab.write(
                os.path.join(
                    outdir,'subsampled_nfw_cat.fits'), format='fits', overwrite=overwrite
                )

        else:
            # No NFW file passed, return None
            nfw_tab = None

        return nfw_tab

    def _nfw_resample_redshift(self, gal_redshifts, nfactor, outdir='.',
                               overwrite=False):
        '''
        Subsample theoretical galaxy redshift distribution to match redshift
        distribution of detected galaxy catalog using a sort of MC rejection
        sampling algorithm

        nfactor: integer multiple of measured catalog size for truth resampling
                 NOTE: In the future, this will be refactored into the number
                       of bootstrap realizations where we keep track of bootstrap
                       index
        '''

        try:
            rng = np.random.default_rng(self.cat_info['nfw_seed'])
        except KeyError:
            rng = np.random.default_rng()

        nfw_file = self.nfw_info['nfw_file']
        nfw = Table.read(nfw_file, format='fits')

        # NOTE: Let's come back to this in the future. By creating Nfactor truth
        #       subsamples w/ len=Ninjections, we can track the variance of the
        #       truth curve as a function of radial bin on top of the mean curve.
        #       However, right now our focus is just the mean truth comparison
        # sample according to number of galaxies injected into the simulations
        # pseudo_nfw = rng.choice(nfw, size=nfactor*self.n_truth_gals, replace=False)
        pseudo_nfw = nfw

        n_selec, bin_edges = np.histogram(
            gal_redshifts, bins=100, range=[gal_redshifts.min(),gal_redshifts.max()]
            )
        n_nfw, bin_edges_nfw = np.histogram(
            pseudo_nfw['redshift'], bins=100, range=[gal_redshifts.min(),gal_redshifts.max()]
            )

        pseudo_prob = n_selec / n_nfw
        domain = np.arange(gal_redshifts.min(), gal_redshifts.max(), 0.0001)

        subsampled_redshifts = []; t = []

        while(len(subsampled_redshifts) < nfactor*len(gal_redshifts)):
            #this_z = rng.choice(nfwstars['redshift'])
            i = rng.choice(len(nfw))
            this_z = nfw[i]['redshift']
            this_bin = np.digitize(this_z, bin_edges_nfw)

            odds = rng.random()
            if (this_bin<len(n_selec)) and (odds <= pseudo_prob[this_bin-1]):
                subsampled_redshifts.append(this_z)
                t.append(nfw[i].as_void())
            else:
                pass

        # nfw_tab is a redshift-resampled subset of the full NFW table
        nfw_tab = Table(np.array(t),names = nfw.colnames)

        # This should be in diagnostics, but do it here for now
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        ax.hist(nfw_tab['redshift'],bins=100,range=[gal_redshifts.min(),\
            gal_redshifts.max()],histtype='step',label='nfw resamp')
        ax.hist(gal_redshifts,bins=100,range=[gal_redshifts.min(),\
            gal_redshifts.max()],histtype='step',label='selected galaxies')
        ax.set_xlabel('Galaxy redshift')
        ax.set_ylabel('Number')
        ax.legend()
        fig.savefig(os.path.join(outdir,'redshift_histograms.png'))

        return nfw_tab

    def _nfw_transform_shear(self, nfw_tab):
        '''
        Repeat shear transform above but with NFW info
        '''

        xc = self.nfw_info['nfw_center'][0]
        yc = self.nfw_info['nfw_center'][1]

        shear_inputs = {
            'x': nfw_tab[self.nfw_info['xy_args'][0]],
            'y': nfw_tab[self.nfw_info['xy_args'][1]],
            'g1': nfw_tab[self.nfw_info['shear_args'][0]],
            'g2': nfw_tab[self.nfw_info['shear_args'][1]]
            }

        shears = ShearCalc(inputs = shear_inputs)
        shears.get_r_gtan(xc=xc, yc=yc)

        # Now populate Annular object attributes with these columns
        x = shears.x ; y = shears.y
        r = shears.r
        gtan = shears.gtan
        gcross = shears.gcross

        # No galaxies in the reference NFW file should have been cut because |g|<0
        assert(len(shears.x) == len (nfw_tab[self.nfw_info['xy_args'][0]]))

        nfw_tab.add_columns([shears.r, shears.gtan, shears.gcross],
                       names=['r', 'gtan', 'gcross']
                    )

        return nfw_tab

    def compute_profile(self, outfile, nfw_tab=None, overwrite=False):
        '''
        Computes mean tangential and cross shear of background (redshift-filtered)
        galaxies in azimuthal bins
        '''
        minrad = self.annular_info['rmin']
        maxrad = self.annular_info['rmax']
        nbins = self.annular_info['nbins']

        bins = np.linspace(minrad, maxrad, nbins)

        counts, bins = np.histogram(self.r, bins=bins)

        N = len(bins) - 1
        midpoint_r = np.zeros(N)
        gtan_mean = np.zeros(N)
        gtan_err = np.zeros(N)
        gcross_mean = np.zeros(N)
        gcross_err = np.zeros(N)

        i = 0
        for b1, b2 in zip(bins[:-1], bins[1:]):

            annulus = (self.r >= b1) & (self.r < b2)
            n = counts[i]
            midpoint_r[i] = np.mean([b1, b2])

            weighted_gtan_stats = DescrStatsW(self.gtan[annulus], weights=self.weight[annulus], ddof=0)
            weighted_gcross_stats = DescrStatsW(self.gcross[annulus], weights=self.weight[annulus], ddof=0)
            
            gtan_mean[i] = weighted_gtan_stats.mean
            gcross_mean[i] = weighted_gcross_stats.mean

            gtan_err[i] = weighted_gtan_stats.std / np.sqrt(n)
            gcross_err[i] =  weighted_gcross_stats.std / np.sqrt(n)

            i += 1
            
        table = Table()
        table.add_columns(
            [counts, midpoint_r, gtan_mean, gcross_mean, gtan_err, gcross_err],
            names=[
                'counts', 'midpoint_r', 'mean_gtan', 'mean_gcross', 'err_gtan', 'err_gcross'
                ],
            )

        # Repeat calculation if an nfw table is supplied, and also compute shear bias

        if nfw_tab is not None:
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

            table.add_columns(
                [nfw_gtan_mean, nfw_gcross_mean, nfw_gtan_err, nfw_gcross_err],
                names=['mean_nfw_gtan', 'mean_nfw_gcross', 'err_nfw_gtan', 'err_nfw_gcross'],
                )

            # Compute shear bias relative to reference NFW
            compute_shear_bias(profile_tab=table)

        print(f'Writing out shear profile catalog to {outfile}')
        table.write(outfile, format='fits', overwrite=overwrite)

        # Print to stdio for quickcheck -- open to better formatting if it exists
        cols = table.colnames
        colstring = '    '.join([col for col in cols])
        print(colstring)

        for i,e in enumerate(table):
            print(f'{e.as_void()}')

        return table

    def plot_profile(self, profile_tab, plotfile, nfw_tab=None):

        if nfw_tab is not None:
            plot_truth = True
        else:
            plot_truth = False

        plotter = ShearProfilePlotter(profile_tab)

        print(f'Plotting shear profile to {plotfile}')
        plotter.plot_tan_profile(outfile=plotfile, plot_truth=plot_truth)

        return

    def run(self, outfile, plotfile, Nresample, overwrite=False):

        outdir = os.path.dirname(outfile)

        # Read in annular catalog
        self.open_table(self.cat_info)

        # Compute gtan/gx
        self.transform_shears(outdir, overwrite=overwrite)

        # Select background galaxies by redshifts
        gal_redshifts = self.redshift_select()

        # Resample NFW file (if supplied) to match galaxy redshift distribution; otherwise return None
        nfw_tab = self.process_nfw(
            gal_redshifts, Nresample, outdir=outdir, overwrite=overwrite
            )

        # Compute azimuthally averaged shear profiles
        profile_tab = self.compute_profile(outfile, nfw_tab, overwrite=overwrite)

        # Plot results
        self.plot_profile(profile_tab, plotfile, nfw_tab=nfw_tab)

        return

