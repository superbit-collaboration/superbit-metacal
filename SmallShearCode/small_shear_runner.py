import numpy as np
import pdb, pudb
from astropy.table import Table, vstack, hstack, join
import glob
import sys, os
from astropy.io import fits
from esutil import htm
from argparse import ArgumentParser

from annular_small_shear import Annular

parser = ArgumentParser()

parser.add_argument('se_file', type=str,
                    help='SExtractor catalog filename')
parser.add_argument('mcal_file', type=str,
                    help='Metacal catalog filename')
parser.add_argument('outfile', type=str,
                    help='Output selected source catalog filename')
parser.add_argument('-run_name', type=str, default=None,
                    help='Name of simulation run')
parser.add_argument('-outdir', type=str, default=None,
                    help='Output directory')
parser.add_argument('-truth_file', type=str, default=None,
                    help='Truth file containing redshifts')
parser.add_argument('-nfw_file', type=str, default=None,
                    help='Theory NFW shear catalog')
parser.add_argument('-rmin', type=float, default=100,
                    help='Starting radius value (in pixels)')
parser.add_argument('-rmax', type=float, default=5200,
                    help='Ending radius value (in pixels)')
parser.add_argument('-nbins', type=int, default=18,
                    help='Number of radial bins')
parser.add_argument('-shear_cutoff', type=float, default=None,
                    help='Maximum gtan to include in shear bias calculation')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='Set to overwrite output files')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Turn on for verbose prints')

class AnnularCatalog():

    """
    This class calculates shear-calibrated ellipticities based on the
    metacal procedure. This will then match against the parent
    SExtractor catalog (set in option in main)
    """

    def __init__(self, cat_info, annular_info):
        """
        cat_info: dict
            A dictionary that must contain the paths for the SExtractor
            catalog, the mcal catalog, and the output catalog filename
        annular_bins: dict
            A dictionary holding the definitions of the annular bins
        """

        self.cat_info = cat_info
        self.annular_info = annular_info
        self.se_file = cat_info['se_file']
        self.mcal_file = cat_info['mcal_file']
        self.outfile = cat_info['mcal_selected']
        self.outdir = cat_info['outdir']
        self.run_name = cat_info['run_name']
        self.truth_file = cat_info['truth_file']
        self.nfw_file = cat_info['nfw_file']

        self.rmin = annular_info['rmin']
        self.rmax = annular_info['rmax']
        self.nbins = annular_info['nbins']
        self.coadd_center = annular_info['coadd_center']

        if self.outdir is not None:
            self.outfile = os.path.join(self.outdir, self.outfile)
        else:
            self.outdir = ''


        self.se_cat = Table.read(self.se_file, hdu=2)
        self.mcal = Table.read(self.mcal_file)
        self.joined = None
        self.joined_gals = None
        self.cluster_redshift = None
        self.selected = None
        self.outcat = None

        self.Nse = len(self.se_cat)
        self.Nmcal = len(self.mcal)

        return


    def go(self, overwrite=False, vb=False):

        # compute tangential shear profile and save outputs
        if self.run_name is not None:
            p = f'{self.run_name}_'
        else:
            p = ''

        cat_info = self.cat_info
        annular_info = self.annular_info

        outfile = os.path.join(self.outdir, f'{p}small_shear_profile_cat.fits')
        plotfile = os.path.join(self.outdir, f'{p}small_shear_profile.png')

        if self.nfw_file is not None:

            nfw_truth_tab = Table.read(self.nfw_file, format='fits')
            nfw_truth_xcenter = nfw_truth_tab.meta['NFW_XCENTER']
            nfw_truth_ycenter = nfw_truth_tab.meta['NFW_YCENTER']

            nfw_info = {
                'nfw_file': self.nfw_file,
                'xy_args': ['x_image','y_image'],
                'shear_args': ['nfw_g1','nfw_g2'],
                'nfw_center': [nfw_truth_xcenter, nfw_truth_ycenter]
                }


        else: nfw_info = None

        # Runs the Annular class in annular_jmac.py
        # Compute cross/tan shear, select background galaxies, obtain shear profile
        # runner = AnnularRunner(cat_info, annular_info)
        annular = Annular(cat_info, annular_info, nfw_info, run_name=self.run_name, vb=vb)

        annular.run(outfile, plotfile, overwrite=overwrite)

        return

def main(args):

    se_file = args.se_file
    mcal_file = args.mcal_file
    run_name = args.run_name
    outfile = args.outfile
    outdir = args.outdir
    truth_file = args.truth_file
    nfw_file = args.nfw_file
    rmin = args.rmin
    rmax = args.rmax
    nbins = args.nbins
    shear_cutoff = args.shear_cutoff
    overwrite = args.overwrite
    vb = args.vb

    # Define position args
    xy_cols = ['X_IMAGE_se', 'Y_IMAGE_se']
    shear_args = ['g1_Rinv', 'g2_Rinv']

    ## Get center of galaxy cluster for fitting
    ## Throw error if image can't be read in

    try:
        coadd_im_name = os.path.join(outdir, f'{run_name}_mock_coadd.sub.fits')
        assert os.path.exists(coadd_im_name) is True
        hdr = fits.getheader(coadd_im_name)
        xcen = hdr['CRPIX1']; ycen = hdr['CRPIX2']
        coadd_center = [xcen, ycen]
        print(f'Read image data and setting image NFW center to ({xcen},{ycen})')

    except Exception as e:
        print('\n\n\nNo coadd image center found, cannot calculate tangential shear\n\n.')
        pdb.set_trace()
        raise e

    ## n.b outfile is the name of the metacalibrated &
    ## quality-selected galaxy catalog

    cat_info={
        'se_file': se_file,
        'mcal_file': mcal_file,
        'run_name': run_name,
        'mcal_selected': outfile,
        'outdir': outdir,
        'truth_file': truth_file,
        'nfw_file': nfw_file
    }

    annular_info = {
        'rmin': rmin,
        'rmax': rmax,
        'nbins': nbins,
        'coadd_center': coadd_center,
        'xy_args': xy_cols,
        'shear_args': shear_args,
        'shear_cutoff': shear_cutoff
    }

    # run everything -- including alpha calc

    annular_cat = AnnularCatalog(cat_info, annular_info)

    annular_cat.go(overwrite=overwrite, vb=vb)

    return 0

if __name__ == '__main__':

    args = parser.parse_args()

    rc = main(args)

    if rc == 0:
        print('make_annular_catalog.py has completed succesfully')
    else:
        print(f'make_annular_catalog.py has failed w/ rc={rc}')
