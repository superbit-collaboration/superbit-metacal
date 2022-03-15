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

parser = ArgumentParser()

# NOTE: This remains for legacy calls. The main way to run the analysis is now with
#       the AnnularRunner class
parser.add_argument('annular_file', type=str,
                    help='Annular catalog filename')
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
# parser.add_argument('outfile', type=str,
#                     help='Output selected source catalog filename')
# parser.add_argument('-run_name', type=str, default=None,
#                     help='Name of simulation run')
# parser.add_argument('-outdir', type=str, default=None,
#                     help='Output directory')
# parser.add_argument('--overwrite', action='store_true', default=False,
#                     help='Set to overwrite output files')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Turn on for verbose prints')

class Annular(object):

    def __init__(self, cat_info=None, annular_info=None, vb=False):

        """
        :incat:   table that will be read in
        :outcat:  name of output shear profile catalog
        :g1/2:    reduced-shear components
        :gcross:  cross-shear (B-mode)
        :gtan:    tangential shear (E-mode)
        :x/y:     x and y positions
        :r:       distance of galaxy to NFW center
        :nbins:   number of radial bins for averaging (default = 5)
        :start/endrad: region to consider
        """

        self.cat_info = cat_info
        self.annular_info = annular_info
        self.vb = vb
        self.g1 = None
        self.g2 = None
        self.gcross = None
        self.gtan = None
        self.x = None
        self.y = None
        self.r = None
        self.mu = None

        """
        self.xc = annular_info['nfw_center'][0]
        self.yc = annular_info['nfw_center'][1]
        self.startrad = annular_info['shear_args'][0]
        self.endrad = annular_info['shear_args'][1]
        self.nbins = annular_info['nbins']
        """

        return

    def open_table(self,cat_info):

        try:
            tab = Table.read(cat_info['incat'], format='fits')
        except:
            tab = Table.read(cat_info['incat'], format='ascii')

        try:
            self.x = tab[cat_info['xy_args'][0]]
            self.y = tab[cat_info['xy_args'][1]]
            self.g1 = tab[cat_info['shear_args'][0]]
            self.g2 = tab[cat_info['shear_args'][1]]
            #self.mu = tab['nfw_mu']
        except:
            print('Could not load xy/g1g2 columns; check supplied column names?')

        return

    def get_r_gtan(self,write=False):
        """
        Populates self.r with radial distance of galaxies from (xc,yc)
        and self.gtan/gcross with tangential and cross ellipticities

        Failed shape measurements with g1/g2 = -999999 get filtered out
        """

        xc = self.annular_info['nfw_center'][0]
        yc = self.annular_info['nfw_center'][1]

        #wg=(self.mu>1)
        g = np.sqrt(self.g1**2 + self.g2**2)
        std_g = np.std(g)
        wg = (g > 0.)
        print(f'## {len(wg.nonzero()[0]} galaxies actually used for calculation')

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

        """
        self.gtan = self.gtan%/(2*np.std(g))
        self.gcross = self.gcross
        """

        if write == True:
            newtab = Table()
            newtab.add_columns(
                [x, y, self.r, self.gtan, self.gcross],
                names=['x', 'y', 'r', 'gcross', 'gtan']
                )
            newtab.write('annular_inputs.fits', format='fits', overwrite=True)

        return

    def do_annular(self):

        """
        workhorse; this will contain the actual azimuthal binning
        """
        minrad = self.annular_info['rad_args'][0]
        maxrad = self.annular_info['rad_args'][1]
        num_bins = self.annular_info['nbins']

        number,bins = np.histogram(self.r, bins=num_bins, range=(minrad, maxrad))

        print("##\n r       n        gtan      err_gtan     gcross     err_gcross")

        for i in range(len(bins)-1):
            annulus = (self.r>=bins[i]) & (self.r<bins[i+1])
            midpoint_r = np.mean([bins[i],bins[i+1]])
            n = number[i]
            gtan_mean = np.mean(self.gtan[annulus])
            gcross_mean = np.mean(self.gcross[annulus])
            gtan_err = np.std(self.gtan[annulus])/np.sqrt(n)
            gcross_err = np.std(self.gcross[annulus])/np.sqrt(n)

            print("%.3f   %d   %f   %f   %f   %f" % (midpoint_r,n,gtan_mean,gtan_err,gcross_mean,gcross_err))

        return


    def run(self, write=False):

        self.open_table(self.cat_info)
        self.get_r_gtan(write=write)
        self.do_annular()

        return

class AnnularRunner(object):
    '''
    Helper class to run the tangential shear profile measurement
    done in the Annular class

    This largely exists as a way of running all `shear_profile` module
    code with only one script call for pipe.py
    '''

    def __init__(self, annular_filename, colnames, r1, r2, Nbins, nfw_cen):
        '''
        annular_filename: str
            Filepath to the desired annular catalog
        colnames: dict
            A dictionary of column definitions for g{1/2}, x, y
        r1, r2: float
            Extremal values for radial bins (in pixels)
        Nbins: np.array
            Number of radial bins to use
        nfw_cen: list, np.array
            list of nfw center in image coords (of coadd image)
        '''

        self.filename = annular_filename
        self.colnames = colnames
        self.r1 = r1
        self.r2 = r2
        self.Nbins = Nbins
        self.nfw_cen

        self.g1_col = colnames['g1']
        self.g2_col = colnames['g2']
        self.x_col = colnames['x']
        self.y_col = colnames['y']

        # create dicts that Annular expects
        cat_info = {
            'incat': self.filename,
            'xy_args': [self.x_col, self.y_col],
            'shear_args': [self.g1_col, self.g2_col]
            }

        annular_info = {
            'rad_args': [self.r1, self.r2],
            'nfw_center': self.nfw_cen,
            'nbins': self.Nbins
            }

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
        print(" python annular_jmac.py incat g1_arg g2_arg start_rad end_rad n_bins\n \n")

    else:
        pass

    # Define catalog args
    incat = args.annular_file
    g1_arg = args.g1_col
    g2_arg = args.g2_col
    startrad = args.start_rad
    endrad = args.end_rad
    num_bins = args.nbins
    vb = args.vb

    # Define annular args
    x_arg = 'X_IMAGE'
    y_arg = 'Y_IMAGE'

    #x_arg = 'x_image'
    #y_arg = 'y_image'
    #startrad = 180
    #endrad = 4000
    nfw_center = [5031, 3353]
    #nfw_center = [4784,3190]

    print_header(args)

    cat_info = {
        'incat': incat,
        'xy_args': [x_arg, y_arg],
        'shear_args': [g1_arg, g2_arg]
        }

    annular_info = {
        'rad_args': [startrad, endrad],
        'nfw_center': nfw_center,
        'nbins': num_bins
        }

    annular = Annular(cat_info, annular_info, vb=vb)
    annular.run(write=True)

    return

if __name__ == '__main__':

    args = parser.parse_args()

    rc = main(args)

    if rc == 0:
        print('annular_jmac.py has completed succesfully')
    else:
        print(f'annular_jmac.py has failed w/ rc={rc}')
