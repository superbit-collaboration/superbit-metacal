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
import pudb, pdb
from esutil import htm

# from superbit_lensing.shear_profiles.shear_plots import ShearProfilePlotter

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
parser.add_argument('-truthfile', type=str, default=None,
                    help='Truth file containing redshift information')
parser.add_argument('-nfwfile', type=str, default=None,
                    help='Theory NFW shear catalog')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Set to overwrite output files')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Turn on for verbose prints')


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

    def get_r_gtan(self,xc,yc):
        '''
        Calculate distance from reference point located at (xc, yc) and rotate
        g1, g2 of galaxies into gtan, gcross

        :xc:   x coordinate of reference point for shear calc
        :yc:   y coordinate of reference point for shear calc
        '''

        g = np.sqrt(self.g1**2 + self.g2**2)
        std_g = np.std(g)
        wg = (g >= 0.)

        nbad = len(wg[wg < 0])
        print(f'## {nbad} of {len(g)} galaxies removed due to |g| < 0')

        self.g1 = self.g1[wg] #*np.sqrt(2)
        self.g2 = self.g2[wg] #*np.sqrt(2)
        self.x = self.x[wg]
        self.y = self.y[wg]

        self.r = np.sqrt(((self.x-xc)**2.) + ((self.y-yc)**2.))
        phi = np.arctan2((self.y-yc), (self.x-xc))

        print(f'## Mean g: {np.mean(g):.3f} sigma_g: {np.std(g):.3f}')

        self.gtan= -1.*(self.g1*np.cos(2.*phi) + self.g2*np.sin(2.*phi))
        # note that annular.c has opposite sign convention
        self.gcross = self.g1*np.sin(2.*phi) - self.g2*np.cos(2.*phi)

        return


class Annular(object):

    def __init__(self, cat_info, annular_info, nfw_info=None, run_name=None, vb=False):

        """
        :infile:   table that will be read in
        :outcat:   name of output shear profile catalog
        :truthfile: name of truth file with redshift info
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
        self.vb = vb
        self.g1 = None
        self.g2 = None
        self.gcross = None
        self.gtan = None
        self.ra = None
        self.dec = None
        self.x = None
        self.y = None
        self.r = None

        return

    def open_table(self, cat_info):

        annular_file = cat_info['infile']
        print(f'loading annular file {annular_file}')

        try:
            tab = Table.read(annular_file, format='fits')
        except:
            tab = Table.read(annular_file, format='ascii')
        try:
            self.x = tab[cat_info['xy_args'][0]]
            self.y = tab[cat_info['xy_args'][1]]
            self.g1 = tab[cat_info['shear_args'][0]]
            self.g2 = tab[cat_info['shear_args'][1]]
            self.ra = tab['ra']
            self.dec = tab['dec']


        except Exception as e:
            print('Could not load xy/g1g2/radec columns; check supplied column names?')
            print(f'cat_info = {cat_info}')
            raise e

        return

    def transform_shears_orig(self, outdir, overwrite=False):
        '''
        Populates self.r with radial distance of galaxies from (xc,yc)
        and self.gtan/gcross with tangential and cross ellipticities

        Failed shape measurements with g1/g2 = -999999 get filtered out

        DEPRECATED
        '''

        xc = self.annular_info['nfw_center'][0]
        yc = self.annular_info['nfw_center'][1]
        g1 = self.g1; g2 = self.g2
        x = self.x; y = self.y

        #wg=(self.mu>1)
        g = np.sqrt(g1**2 + g2**2)
        std_g = np.std(g)
        wg = (g >= 0.)

        nbad = len(wg[wg < 0])
        print(f'## {nbad} of {len(g)} galaxies removed due to |g| < 0')

        g1 = self.g1[wg] #*np.sqrt(2)
        g2 = self.g2[wg] #*np.sqrt(2)
        x = self.x[wg]
        y = self.y[wg]

        self.r = np.sqrt(((x-xc)**2.) + ((y-yc)**2.))

        phi = np.arctan2((y-yc), (x-xc))
        print(f'## Mean g: {np.mean(g):.3f} sigma_g: {np.std(g):.3f}')
        self.gtan= -1.*(g1*np.cos(2.*phi) + g2*np.sin(2.*phi))

        # note that annular.c has opposite sign convention
        self.gcross = g1*np.sin(2.*phi) - g2*np.cos(2.*phi)

        newtab = Table()
        newtab.add_columns(
            [x, y, self.r, self.gtan, self.gcross],
            names=['x', 'y', 'r', 'gcross', 'gtan']
            )

        outfile = os.path.join(outdir, 'transformed_shear_tab.fits')
        newtab.write(outfile, format='fits', overwrite=overwrite)

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

        xc = self.annular_info['nfw_center'][0]
        yc = self.annular_info['nfw_center'][1]

        shears = ShearCalc(inputs = shear_inputs)
        shears.get_r_gtan(xc=xc, yc=yc)

        # Now populate Annular object attributes with these columns
        x = shears.x ; y = shears.y
        self.r = shears.r
        self.gtan = shears.gtan
        self.gcross = shears.gcross

        newtab = Table()
        newtab.add_columns(
            [x, y, self.r, self.gtan, self.gcross],
            names=['x', 'y', 'r', 'gcross', 'gtan']
            )

        outfile = os.path.join(outdir, 'transformed_shear_tab.fits')
        newtab.write(outfile, format='fits', overwrite=overwrite)

        return

    def redshift_select(self):
        '''
        Select background galaxies from larger transformed shear catalog:
            - Load in truth file
            - Select background galaxies behind galaxy cluster
            - Match in RA/Dec to transformed shear catalog
            - Filter self.r, self.gtan, self.gcross to be background-only
        '''

        truthfile = self.cat_info['truthfile']

        try:
            truth = Table.read(truthfile)
            if self.vb is True:
                print(f'Read in truth file {truthfile}')

        except FileNotFoundError as fnf_err:
            print(f'truth catalog {truthfile} not found, check name/type?')
            raise fnf_err

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
                                            radius = 5E-4
                                            )

        print(f'# {len(dist)} of {len(self.ra)} objects matched to truth background galaxies')

        self.gtan = self.gtan[ann_file_ind]
        self.gcross = self.gcross[ann_file_ind]
        self.r = self.r[ann_file_ind]
        gal_redshifts = truth_bg_gals[truth_bg_ind]['redshift']

        return gal_redshifts

    def process_nfw(self, gal_redshifts):
        '''
        Subsample theoretical galaxy redshift distribution to match redshift
        distribution of detected galaxy catalog using a sort of MC rejection
        sampling algorithm

        Also compute g_tan, g_cross
        '''
        pass
        if self.nfw_info is not None:

            self._nfw_resample(gal_redshifts)

            tt.write('subsampled_nfw_cat.fits',format='fits')

        else:
            # No NFW file passed, return None
            tt = None

        return tt

    def _nfw_resample(self, gal_redshifts,outdir='.',overwrite=False):
        '''
        Subsample theoretical galaxy redshift distribution to match redshift
        distribution of detected galaxy catalog using a sort of MC rejection
        sampling algorithm
        '''

        rng = np.random.default_rng()

        nfwfile = self.nfw_info['nfwfile']
        nfw = Table.read(nfwfile,format='fits')

        # 34,300 galaxies injected into the simulations
        pseudo_nfw = rng.choice(nfw, size=34300, replace=False)

        n_selec,bin_edges=np.histogram(gal_redshifts,bins=100,\
            range=[gal_redshifts.min(),gal_redshifts.max()])
        n_nfw,bin_edges_nfw=np.histogram(pseudo_nfw['redshift'],bins=100,\
            range=[gal_redshifts.min(),gal_redshifts.max()])

        pseudo_prob = n_selec/n_nfw
        domain = np.arange(gal_redshifts.min(),gal_redshifts.max(),0.0001)

        subsampled_redshifts = []; t = []

        while(len(subsampled_redshifts) < len(gal_redshifts)):
            #this_z = rng.choice(nfwstars['redshift'])
            i = rng.choice(len(nfw))
            this_z = nfw[i]['redshift']
            this_bin = np.digitize(this_z,bin_edges_nfw)

            odds = rng.random()
            if (this_bin<len(n_selec)) and (odds <= pseudo_prob[this_bin-1]):
                subsampled_redshifts.append(this_z)
                t.append(nfw[i].as_void())
            else:
                pass

        tt = Table(np.array(t),names = nfw.colnames)
        tt.write(os.path.join(outdir,'subsampled_nfw_cat.fits'),format='fits', overwrite=overwrite)

        # This should be a diagnostic, but do it here for now
        fig,ax=plt.subplots(1,1,figsize=(8,6))

        ax.hist(tt['redshift'],bins=100,range=[gal_redshifts.min(),\
            gal_redshifts.max()],histtype='step',label='nfw resamp')
        ax.hist(gal_redshifts,bins=100,range=[gal_redshifts.min(),\
            gal_redshifts.max()],histtype='step',label='selected galaxies')
        ax.set_xlabel('Galaxy redshift')
        ax.set_ylabel('Number')
        ax.legend()
        fig.savefig(os.path.join(outdir,'redshift_histograms.png'))

        return tt

    def _nfw_transform(self, tt):
        '''
        Repeat shear transform above but with NFW info
        '''

        pass
        xc = self.nfw_info['nfw_center'][0]
        yc = self.nfw_info['nfw_center'][1]
        x = nfw_table[self.nfw_info['xy_args'][0]]
        y = nfw_table[self.nfw_info['xy_args'][1]]
        g1 = nfw_table[self.nfw_info['shear_args'][0]]
        g2 = nfw_table[self.nfw_info['shear_args'][1]]


    def compute_profile(self, outfile, nfwtab=None, overwrite=False):
        '''
        Computes mean tangential and cross shear of background (redshift-filtered)
        galaxies in azimuthal bins
        '''

        minrad = self.annular_info['rad_args'][0]
        maxrad = self.annular_info['rad_args'][1]
        num_bins = self.annular_info['nbins']

        bins = np.linspace(minrad, maxrad, num_bins)

        counts, bins = np.histogram(self.r, bins=bins)

        print("##\n r       n        gtan      err_gtan     gcross     err_gcross")

        N = len(bins) - 1
        midpoint_r = np.zeros(N)
        gtan_mean = np.zeros(N)
        gtan_err = np.zeros(N)
        gcross_mean = np.zeros(N)
        gcross_err = np.zeros(N)

        # for i in range(len(bins)-1):
        i = 0
        for b1, b2 in zip(bins[:-1], bins[1:]):
            # annulus = (self.r>=bins[i]) & (self.r<bins[i+1])
            annulus = (self.r >= b1) & (self.r < b2)
            n = counts[i]
            midpoint_r[i] = np.mean([b1, b2])
            gtan_mean[i] = np.mean(self.gtan[annulus])
            gcross_mean[i] = np.mean(self.gcross[annulus])
            gtan_err[i] = np.std(self.gtan[annulus]) / np.sqrt(n)
            gcross_err[i] = np.std(self.gcross[annulus]) / np.sqrt(n)

            print("%.3f   %d   %f   %f   %f   %f" %\
                  (midpoint_r[i], n, gtan_mean[i], gtan_err[i], gcross_mean[i], gcross_err[i])
                  )

            i += 1

        table = Table()
        table.add_columns(
            [counts, midpoint_r, gtan_mean, gcross_mean, gtan_err, gcross_err],
            names=[
                'counts', 'midpoint_r', 'gtan_mean', 'gcross_mean', 'gtan_err', 'gcross_err'
                ],
            )

        table.write(outfile, format='fits', overwrite=overwrite)

        return

    def plot_profile(self, cat_file, truthfile, plot_file):

        # plotter = ShearProfilePlotter(cat_file, truthfile)
        # plotter.plot(plot_file)

        return

    def run(self, outfile, plotfile, overwrite=False):

        outdir = os.path.dirname(outfile)

        # Read in annular catalog
        self.open_table(self.cat_info)

        # Compute gtan/gx
        self.transform_shears(outdir, overwrite=overwrite)

        # Select background galaxies using reshifts
        gal_redshifts = self.redshift_select()

        # Resample reference NFW file to match redshift distribution of galaxies
        nfw_shear_resamp = self._nfw_resample(gal_redshifts, outdir=outdir, overwrite=overwrite)

        # Compute azimuthally averaged shear profiles
        self.compute_profile(outfile, overwrite=overwrite)

        # plotting function stil needs to be refactored...
        # self.plot_profile(plotfile)

        return

def print_header(args):

    print('###\n### Shear profile calculation\n###')
    print('### created with:')
    print(f'### {args}\n###')
    print('## r: midpoint radius of annulus')
    print('## n: number of objects for shear calculation')
    print('## gtan: tangential (E-mode) reduced-shear')
    print('## err_gtan: standard error of gtan')
    print('## gcross: cross (B-mode) reduced-shear')
    print('## err_gcross: standard error of gcross')

    return

def main(args):

    """
    NOTE: This remains for legacy calls. The main way to run the analysis is now with
    the AnnularRunner class

    TODO: clean up input arguments, using -o --option format
    and make startrad/endrad options as well, rather than magic numbers
    """

    if ( ( len(args) < 3) or (args == '-h') or (args == '--help') ):
        print("\n### \n### annular_jmac is a routine which takes an ascii table and x/y/g1/g2 column names as its input and returns shear profiles")
        print("### Note that at present, annular_jmac assumes that the shear arguments [g1, g2] are *reduced shear*, not image ellipticities\n###\n")
        print(" python annular_jmac.py infile g1_arg g2_arg start_rad end_rad n_bins\n \n")

    else:
        pass

    # Define catalog args
    infile = args.annular_file
    outfile = args.outfile
    g1_arg = args.g1_col
    g2_arg = args.g2_col
    startrad = args.start_rad
    endrad = args.end_rad
    num_bins = args.nbins
    outdir = args.outdir
    run_name = args.run_name
    overwrite = args.overwrite
    vb = args.vb

    if outdir is not None:
        outfile = os.path.join(self.outdir, outfile)

    """
    # If obtaining "true" g_tan using the truth file:
    x_arg = 'x_image'
    y_arg = 'y_image'
    startrad = 100
    endrad = 5000
    nfw_center = [4784,3190]
    """

    # Otherwise, define annular args
    x_arg = 'X_IMAGE'
    y_arg = 'Y_IMAGE'
    nfw_center = [5031, 3353]

    print_header(args)

    cat_info = {
        'infile': infile,
        'xy_args': [x_arg, y_arg],
        'shear_args': [g1_arg, g2_arg]
        }

    annular_info = {
        'rad_args': [startrad, endrad],
        'nfw_center': nfw_center,
        'nbins': num_bins
        }

    annular = Annular(
        cat_info, annular_info, nfw_info, run_name=run_name, vb=vb
        )

    annular.run(outfile, overwrite=overwrite, outdir=outdir)

    return

if __name__ == '__main__':

    args = parser.parse_args()

    rc = main(args)

    if rc == 0:
        print('annular_jmac.py has completed succesfully')
    else:
        print(f'annular_jmac.py has failed w/ rc={rc}')
