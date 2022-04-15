import numpy as np
import pdb, pudb
from astropy.table import Table, vstack, hstack, join
import glob
import sys, os
from astropy.io import fits
from esutil import htm
from argparse import ArgumentParser

from annular_jmac import Annular

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

    def __init__(self, cat_info, annular_bins):
        """
        cat_info: dict
            A dictionary that must contain the paths for the SExtractor
            catalog, the mcal catalog, and the output catalog filename
        annular_bins: dict
            A dictionary holding the definitions of the annular bins
        """

        self.se_file = cat_info['se_file']
        self.mcal_file = cat_info['mcal_file']
        self.outfile = cat_info['outfile']
        self.outdir = cat_info['outdir']
        self.run_name = cat_info['run_name']
        self.truth_file = cat_info['truth_file']
        self.nfw_file = cat_info['nfw_file']

        self.rmin = annular_bins['rmin']
        self.rmax = annular_bins['rmax']
        self.nbins = annular_bins['nbins']

        if self.outdir is not None:
            self.outfile = os.path.join(self.outdir, self.outfile)
        else:
            self.outdir = ''

        self.se_cat = Table.read(self.se_file, hdu=2)
        self.mcal = Table.read(self.mcal_file)
        self.joined = None
        self.selected = None
        self.outcat = None

        self.Nse = len(self.se_cat)
        self.Nmcal = len(self.mcal)

        return

    def join(self, overwrite=False):

        # rename a few cols for joining purposes
        colmap = {
            'ALPHAWIN_J2000': 'ra',
            'DELTAWIN_J2000': 'dec',
            'NUMBER': 'id'
        }
        for old, new in colmap.items():
            self.se_cat.rename_column(old, new)

        # TODO: Should we remove a few duplicate cols here?

        self.joined = join(
            self.se_cat, self.mcal, join_type='inner',
            keys=['id', 'ra', 'dec'], table_names=['se', 'mcal']
            )

        Nmin = min([self.Nse, self.Nmcal])
        Nobjs = len(self.joined)

        if (Nobjs != Nmin):
            raise ValueError('There was an error while joining the SE and mcal ' +
                             f'catalogs;' +
                             f'\nlen(SE)={self.Nse}' +
                             f'\nlen(mcal)={self.Nmcal}' +
                             f'\nlen(joined)={Nobjs}'
                             )

        print(f'{Nobjs} mcal objects joined to reference SExtractor cat')

        try:
            # save full catalog to file
            if self.run_name is None:
                p = ''
            else:
                p = f'{self.run_name}_'

            outfile = os.path.join(self.outdir, f'{p}full_joined_catalog.fits')
            self.joined.write(outfile, overwrite=overwrite)

        # we *want* it to fail loudly!
        except OSError as err:
            print('Cannot overwrite {outfile} unless `overwrite` is set to True!')
            raise err

        return

    def make_table(self, overwrite=False):
        """
        - Select from catalog on g_cov, T/T_psf, etc.
        - Correct g1/g2_noshear for the Rinv quantity (see Huff & Mandelbaum 2017)
        - Save shear-response corrected ellipticities to an output table
        """

        # This step both applies selection cuts and generates shear-calibrated
        # tangential ellipticity moments

        qualcuts = self._compute_metacal_quantities()

        # I would love to be able to save qualcuts as comments in FITS header
        # but can't figure it out rn
        # self.selected.meta={qualcuts}

        #self.selected.write('selected_metacal_bgCat.fits',overwrite=True)
        self.selected.write(self.outfile, format='fits', overwrite=overwrite)

        return

    def _compute_metacal_quantities(self):
        """
        - Cut sources on S/N and minimum size (adapted from DES cuts).
        - compute mean r11 and r22 for galaxies: responsivity & selection
        - divide "no shear" g1/g2 by r11 and r22, and return

        TO DO: make the list of cuts an external config file

        """

        # TODO: we should allow for this to be a config option
        mcal_shear = 0.01

        # TODO: It would be nice to move selection cuts
        # to a different file
        min_Tpsf = 1. # orig 1.15
        max_sn = 1000
        min_sn = 10 # orig 8 for ensemble
        min_T = 0.03 # orig 0.05
        max_T = 10 # orig inf
        covcut = 1E-2 # orig 1 for ensemble

        qualcuts = {'min_Tpsf':min_Tpsf, 'max_sn':max_sn, 'min_sn':min_sn,
                    'min_T':min_T, 'max_T':max_T, 'covcut':covcut}

        print(f'#\n# cuts applied: Tpsf_ratio>{min_Tpsf:.2f}' +\
              f' SN>{min_sn:.1f} T>{min_T:.2f} covcut={covcut:.1e}\n#\n')

        noshear_selection = self.mcal[(self.mcal['T_noshear']>=min_Tpsf*self.mcal['Tpsf_noshear'])\
                                        & (self.mcal['T_noshear']<max_T)\
                                        & (self.mcal['T_noshear']>=min_T)\
                                        & (self.mcal['s2n_r_noshear']>min_sn)\
                                        & (self.mcal['s2n_r_noshear']<max_sn)\
                                        & (self.mcal['g_cov_noshear'][:,0,0]<covcut)\
                                        & (self.mcal['g_cov_noshear'][:,1,1]<covcut)
                                       ]

        selection_1p = self.mcal[(self.mcal['T_1p']>=min_Tpsf*self.mcal['Tpsf_1p'])\
                                      & (self.mcal['T_1p']<=max_T)\
                                      & (self.mcal['T_1p']>=min_T)\
                                      & (self.mcal['s2n_r_1p']>min_sn)\
                                      & (self.mcal['s2n_r_1p']<max_sn)\
                                      & (self.mcal['g_cov_1p'][:,0,0]<covcut)\
                                      & (self.mcal['g_cov_1p'][:,1,1]<covcut)
                                  ]

        selection_1m = self.mcal[(self.mcal['T_1m']>=min_Tpsf*self.mcal['Tpsf_1m'])\
                                      & (self.mcal['T_1m']<=max_T)\
                                      & (self.mcal['T_1m']>=min_T)\
                                      & (self.mcal['s2n_r_1m']>min_sn)\
                                      & (self.mcal['s2n_r_1m']<max_sn)\
                                      & (self.mcal['g_cov_1m'][:,0,0]<covcut)\
                                      & (self.mcal['g_cov_1m'][:,1,1]<covcut)
                                  ]

        selection_2p = self.mcal[(self.mcal['T_2p']>=min_Tpsf*self.mcal['Tpsf_2p'])\
                                      & (self.mcal['T_2p']<=max_T)\
                                      & (self.mcal['T_2p']>=min_T)\
                                      & (self.mcal['s2n_r_2p']>min_sn)\
                                      & (self.mcal['s2n_r_2p']<max_sn)\
                                      & (self.mcal['g_cov_2p'][:,0,0]<covcut)\
                                      & (self.mcal['g_cov_2p'][:,1,1]<covcut)
                                  ]

        selection_2m = self.mcal[(self.mcal['T_2m']>=min_Tpsf*self.mcal['Tpsf_2m'])\
                                      & (self.mcal['T_2m']<=max_T)\
                                      & (self.mcal['T_2m']>=min_T)\
                                      & (self.mcal['s2n_r_2m']>min_sn)\
                                      & (self.mcal['s2n_2m']<max_sn)\
                                      & (self.mcal['g_cov_2m'][:,0,0]<covcut)\
                                      & (self.mcal['g_cov_2m'][:,1,1]<covcut)
                                  ]

        # assuming delta_shear in ngmix_fit_superbit is 0.01
        r11_gamma = (np.mean(noshear_selection['g_1p'][:,0]) -
                     np.mean(noshear_selection['g_1m'][:,0])) / (2.*mcal_shear)
        r22_gamma = (np.mean(noshear_selection['g_2p'][:,1]) -
                     np.mean(noshear_selection['g_2m'][:,1])) / (2.*mcal_shear)

        # assuming delta_shear in ngmix_fit_superbit is 0.01
        r11_S = (np.mean(selection_1p['g_noshear'][:,0]) -
                 np.mean(selection_1m['g_noshear'][:,0])) / (2.*mcal_shear)
        r22_S = (np.mean(selection_2p['g_noshear'][:,1]) -
                 np.mean(selection_2m['g_noshear'][:,1])) / (2.*mcal_shear)

        print(f'# mean values <r11_gamma> = {r11_gamma} ' +\
              f'<r22_gamma> = {r22_gamma}')
        print(f'# mean values <r11_S> = {r11_S} ' +\
              f'<r22_S> = {r22_S}')
        print(f'{len(noshear_selection)} objects passed selection criteria')

        # Populate the selCat attribute with "noshear"-selected catalog
        self.selected = noshear_selection

        # compute noise; not entirely sure whether there needs to be a factor of 0.5 on tot_covar...
        # seems like not if I'm applying it just to tangential ellip, yes if it's being applied to each
        shape_noise = np.std(np.sqrt(self.mcal['g_noshear'][:,0]**2 + self.mcal['g_noshear'][:,1]**2))
        print(f'shape noise is {shape_noise}')

        #shape_noise = 0.3
        tot_covar = shape_noise +\
                self.selected['g_cov_noshear'][:,0,0] +\
                self.selected['g_cov_noshear'][:,1,1]
        weight = 1. / tot_covar

        try:
            r11 = ( noshear_selection['g_1p'][:,0] - noshear_selection['g_1m'][:,0] ) / (2.*mcal_shear)
            r12 = ( noshear_selection['g_2p'][:,0] - noshear_selection['g_2m'][:,0] ) / (2.*mcal_shear)
            r21 = ( noshear_selection['g_1p'][:,1] - noshear_selection['g_1m'][:,1] ) / (2.*mcal_shear)
            r22 = ( noshear_selection['g_2p'][:,1] - noshear_selection['g_2m'][:,1] ) / (2.*mcal_shear)

            #---------------------------------
            # Now add value-adds to table
            self.selected.add_columns(
                [r11, r12, r21, r22],
                names=['r11', 'r12', 'r21', 'r22']
                )

        except ValueError as e:
            # In some cases, these cols are already computed
            print('WARNING: mcal r{ij} value-added cols not added; ' +\
                  'already present in catalog')

        try:
            R = np.array([[r11, r12], [r21, r22]])
            g1_MC = np.zeros_like(r11)
            g2_MC = np.zeros_like(r22)

            N = len(g1_MC)
            for k in range(N):
                Rinv = np.linalg.inv(R[:,:,k])
                gMC = np.dot(Rinv, noshear_selection[k]['g_noshear'])
                g1_MC[k] = gMC[0]
                g2_MC[k] = gMC[1]

            self.selected.add_columns(
                [g1_MC, g2_MC],
                names = ['g1_MC', 'g2_MC']
            )

        except ValueError as e:
            # In some cases, these cols are already computed
            print('WARNING: mcal g{1/2}_MC value-added cols not added; ' +\
                  'already present in catalog')

        self.selected['g1_Rinv'] = self.selected['g_noshear'][:,0]/(r11_gamma + r11_S)
        self.selected['g2_Rinv'] = self.selected['g_noshear'][:,1]/(r22_gamma + r22_S)
        self.selected['R11_S'] = r11_S
        self.selected['R22_S'] = r22_S
        self.selected['weight'] = weight

        return qualcuts

    def compute_tan_shear_profile(self, outfile, plotfile, overwrite=False, vb=False,
                                  xy_cols=['X_IMAGE', 'Y_IMAGE'],
                                  g_cols=['g1_Rinv', 'g2_Rinv'],
                                  nfw_center=[5031, 3353]):

        if self.truth_file is None:
            truth_name = ''.join([self.run_name,'_truth.fits'])
            truth_dir = self.outdir
            truth_file = os.path.join(truth_dir,truth_name)

        else:
            truth_file = self.truth_file

        cat_info = {
            'infile': self.outfile,
            'truth_file': truth_file,
            'nfw_file': self.nfw_file,
            'xy_args': xy_cols,
            'shear_args': g_cols
        }
        annular_info = {
            'rad_args': [self.rmin, self.rmax],
            'nfw_center': nfw_center,
            'nbins': self.nbins
        }

        if self.nfw_file is not None:

            nfw_info = {
                'nfw_file': self.nfw_file,
                'xy_args': ['x_image','y_image'],
                'shear_args': ['nfw_g1','nfw_g2'],
                'nfw_center': [4784, 3190],
            }

        else: nfw_info = None

        # Runs the Annular class in annular_jmac.py
        # Compute cross/tan shear, select background galaxies, obtain shear profile
        # runner = AnnularRunner(cat_info, annular_info)
        annular = Annular(cat_info, annular_info, nfw_info, run_name=self.run_name, vb=vb)

        annular.run(outfile, plotfile, overwrite=overwrite)

        return

    def run(self, overwrite=False, vb=False):

        # match master metacal catalog to source extractor cat
        self.join(overwrite=overwrite)

        # source selection; saves table to self.outfile
        self.make_table(overwrite=overwrite)

        # compute tangential shear profile and save outputs
        if self.run_name is not None:
            p = f'{self.run_name}_'
        else:
            p = ''

        outfile = os.path.join(self.outdir, f'{p}shear_profile_cat.fits')
        plotfile = os.path.join(self.outdir, f'{p}shear_profile.pdf')

        self.compute_tan_shear_profile(outfile, plotfile, overwrite=overwrite, vb=vb)

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
    overwrite = args.overwrite
    vb = args.vb

    cat_info={
        'se_file': se_file,
        'mcal_file': mcal_file,
        'run_name': run_name,
        'outfile': outfile,
        'outdir': outdir,
        'truth_file': truth_file,
        'nfw_file': nfw_file
        }

    annular_bins = {
        'rmin': rmin,
        'rmax': rmax,
        'nbins': nbins
    }

    annular = AnnularCatalog(cat_info, annular_bins)

    # run everything
    annular.run(overwrite=overwrite, vb=vb)

    return 0

if __name__ == '__main__':

    args = parser.parse_args()

    rc = main(args)

    if rc == 0:
        print('make_annular_catalog.py has completed succesfully')
    else:
        print(f'make_annular_catalog.py has failed w/ rc={rc}')
