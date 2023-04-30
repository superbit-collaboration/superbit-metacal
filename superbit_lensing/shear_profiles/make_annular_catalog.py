import numpy as np
import ipdb
from astropy.table import Table, vstack, hstack, join
import glob
import sys, os
from astropy.io import fits
from esutil import htm
from argparse import ArgumentParser

from annular_jmac import Annular, ShearCalc
from superbit_lensing import utils

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('data_dir', type=str,
                        help='Path to cluster data')
    parser.add_argument('run_name', type=str,
                        help='Run name (target name for real data)')
    parser.add_argument('mcal_file', type=str,
                        help='Metacal catalog filename')
    parser.add_argument('outfile', type=str,
                        help='Output selected source catalog filename')
    parser.add_argument('-outdir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('-truth_file', type=str, default=None,
                        help='Truth file containing redshifts')
    parser.add_argument('-nfw_file', type=str, default=None,
                        help='Theory NFW shear catalog')
    parser.add_argument('-Nresample', type=int, default=10,
                        help='The number of NFW redshift resamples to compute')
    parser.add_argument('-rmin', type=float, default=100,
                        help='Starting radius value (in pixels)')
    parser.add_argument('-rmax', type=float, default=5200,
                        help='Ending radius value (in pixels)')
    parser.add_argument('-nfw_seed', type=int, default=None,
                        help='Seed for nfw redshift resampling')
    parser.add_argument('-nbins', type=int, default=18,
                        help='Number of radial bins')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='Set to overwrite output files')
    parser.add_argument('--vb', action='store_true', default=False,
                        help='Turn on for verbose prints')

    return parser.parse_args()

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
        self.detect_cat = cat_info['detect_cat']
        self.mcal_file = cat_info['mcal_file']
        self.outfile = cat_info['mcal_selected']
        self.outdir = cat_info['outdir']
        self.run_name = cat_info['run_name']
        self.truth_file = cat_info['truth_file']
        self.nfw_file = cat_info['nfw_file']
        self.Nresample = cat_info['Nresample']

        self.rmin = annular_info['rmin']
        self.rmax = annular_info['rmax']
        self.nbins = annular_info['nbins']
        self.coadd_center = annular_info['coadd_center']

        if self.outdir is not None:
            self.outfile = os.path.join(self.outdir, self.outfile)
        else:
            self.outdir = ''

        self.se_cat = Table.read(self.detect_cat, hdu=2)
        self.mcal = Table.read(self.mcal_file)
        self.joined = None
        self.joined_gals = None
        self.cluster_redshift = None
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


    def _redshift_select(self, truth_file, overwrite=False):
        '''
        Select background galaxies from larger transformed shear catalog:
            - Load in truth file
            - Select background galaxies behind galaxy cluster
            - Match in RA/Dec to transformed shear catalog
            - Filter self.r, self.gtan, self.gcross to be background-only
            - Also store the number of galaxies injected into simulation
        '''

        joined_cat = self.joined

        try:
            truth = Table.read(truth_file, format='fits')
            #if vb is True:
            print(f'Read in truth file {truth_file}')

        except FileNotFoundError as fnf_err:
            print(f'truth catalog {truth_file} not found, check name/type?')
            raise fnf_err

        truth_gals = truth[truth['obj_class'] == 'gal']
        self.n_truth_gals = len(truth_gals)

        cluster_gals = truth[truth['obj_class']=='cluster_gal']
        cluster_redshift = np.mean(cluster_gals['redshift'])
        self.cluster_redshift = cluster_redshift

        truth_matcher = htm.Matcher(16,
                                        ra = truth_gals['ra'],
                                        dec = truth_gals['dec']
                                        )

        joined_file_ind, truth_ind, dist = truth_matcher.match(
                                            ra = joined_cat['ra'],
                                            dec = joined_cat['dec'],
                                            maxmatch = 1,
                                            radius = 1./3600.
                                            )

        print(f"# {len(dist)} of {len(joined_cat['ra'])} objects matched to truth galaxies")

        gals_joined_cat = joined_cat[joined_file_ind]
        gals_joined_cat.add_column(truth_gals['redshift'][truth_ind])

        try:
            if self.run_name is None:
                p = ''
            else:
                p = f'{self.run_name}_'

                outfile = os.path.join(self.outdir, f'{p}gals_joined_catalog.fits')
                gals_joined_cat.write(outfile, overwrite=overwrite)

        except OSError as err:
            print('Cannot overwrite {outfile} unless `overwrite` is set to True!')
            raise err

        self.joined_gals = gals_joined_cat

        return

    def make_table(self, overwrite=False):
        """
        - Remove foreground galaxies from sample using redshift info in truth file
        - Select from catalog on g_cov, T/T_psf, etc.
        - Correct g1/g2_noshear for the Rinv quantity (see Huff & Mandelbaum 2017)
        - Save shear-response corrected ellipticities to an output table
        """

        # Access truth file name
        cat_info = self.cat_info

        if cat_info['truth_file'] is None:

            truth_name = ''.join([self.run_name,'_truth.fits'])
            truth_dir = self.outdir
            truth_file = os.path.join(truth_dir,truth_name)
            self.cat_info['truth_file'] = truth_file

        else:
            truth_file = self.truth_file

        # Filter out foreground galaxies using redshifts in truth file
        self._redshift_select(truth_file, overwrite=overwrite)

        # Apply selection cuts and produce responsivity-corrected shear moments
        # Return selection (quality) cuts
        qualcuts = self._compute_metacal_quantities()

        # Save selected galaxies to file
        for key in qualcuts.keys():
            self.selected.meta[key] = qualcuts[key]

        self.selected.write(self.outfile, format='fits', overwrite=overwrite)

        return qualcuts

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
        min_Tpsf = 1.0
        max_sn = 1000
        min_sn = 10
        min_T = 0.0
        max_T = 10

        if self.cluster_redshift is not None:
            min_redshift = self.cluster_redshift
        else:
            min_redshift = 0

        print(f'#\n# cuts applied: Tpsf_ratio>{min_Tpsf:.2f}' +\
              f' SN>{min_sn:.1f} T>{min_T:.2f} redshift={min_redshift:.3f}\n#\n')

        qualcuts = {'min_Tpsf' :min_Tpsf,
                    'max_sn' : max_sn,
                    'min_sn' : min_sn,
                    'min_T' : min_T,
                    'max_T' : max_T,
                    'min_redshift' : min_redshift,
                    }

        mcal = self.joined_gals

        noshear_selection = mcal[(mcal['T_r_noshear']>=min_Tpsf*mcal['Tpsf_noshear'])\
                                 & (mcal['T_r_noshear']<max_T)\
                                 & (mcal['T_r_noshear']>=min_T)\
                                 & (mcal['s2n_r_noshear']>min_sn)\
                                 & (mcal['s2n_r_noshear']<max_sn)\
                                 & (mcal['redshift'] > min_redshift)
                                 ]

        selection_1p = mcal[(mcal['T_r_1p']>=min_Tpsf*mcal['Tpsf_1p'])\
                            & (mcal['T_r_1p']<=max_T)\
                            & (mcal['T_r_1p']>=min_T)\
                            & (mcal['s2n_r_1p']>min_sn)\
                            & (mcal['s2n_r_1p']<max_sn)\
                            & (mcal['redshift'] > min_redshift)
                            ]

        selection_1m = mcal[(mcal['T_r_1m']>=min_Tpsf*mcal['Tpsf_1m'])\
                            & (mcal['T_r_1m']<=max_T)\
                            & (mcal['T_r_1m']>=min_T)\
                            & (mcal['s2n_r_1m']>min_sn)\
                            & (mcal['s2n_r_1m']<max_sn)\
                            & (mcal['redshift'] > min_redshift)
                            ]

        selection_2p = mcal[(mcal['T_r_2p']>=min_Tpsf*mcal['Tpsf_2p'])\
                            & (mcal['T_r_2p']<=max_T)\
                            & (mcal['T_r_2p']>=min_T)\
                            & (mcal['s2n_r_2p']>min_sn)\
                            & (mcal['s2n_r_2p']<max_sn)\
                            & (mcal['redshift'] > min_redshift)
                            ]

        selection_2m = mcal[(mcal['T_r_2m']>=min_Tpsf*mcal['Tpsf_2m'])\
                            & (mcal['T_r_2m']<=max_T)\
                            & (mcal['T_r_2m']>=min_T)\
                            & (mcal['s2n_r_2m']>min_sn)\
                            & (mcal['s2n_2m']<max_sn)\
                            & (mcal['redshift'] > min_redshift)
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
        #shape_noise = np.std(np.sqrt(self.mcal['g_noshear'][:,0]**2 + self.mcal['g_noshear'][:,1]**2))

        shape_noise = 0.26

        print(f'shape noise is {shape_noise}')

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

    def compute_tan_shear_profile(self, outfile, plotfile, Nresample,
                                  overwrite=False, vb=False):

        cat_info = self.cat_info

        annular_info = self.annular_info


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

        else:
            nfw_info = None

        # Runs the Annular class in annular_jmac.py
        # Compute cross/tan shear, & obtain single-realization shear profile
        annular = Annular(
            cat_info, annular_info, nfw_info, run_name=self.run_name, vb=vb
            )

        annular.run(outfile, plotfile, Nresample, overwrite=overwrite)

        return

    def run(self, overwrite, vb=False):


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
        Nresample = self.Nresample

        self.compute_tan_shear_profile(
            outfile, plotfile, Nresample, overwrite=overwrite, vb=vb
            )

        return

def main(args):

    data_dir = args.data_dir
    target_name = args.run_name
    mcal_file = args.mcal_file
    outfile = args.outfile
    outdir = args.outdir
    truth_file = args.truth_file
    nfw_file = args.nfw_file
    Nresample = args.Nresample
    rmin = args.rmin
    rmax = args.rmax
    nfw_seed = args.nfw_seed
    nbins = args.nbins
    overwrite = args.overwrite
    vb = args.vb


    # Define position args
    xy_cols = ['X_IMAGE_se', 'Y_IMAGE_se']
    shear_args = ['g1_Rinv', 'g2_Rinv']

    ## Get center of galaxy cluster for fitting
    ## Throw error if image can't be read in

    detect_cat = os.path.join(data_dir, target_name,
                                f'det/cat/{target_name}_coadd_det_cat.fits'
                                )
    detect_im = os.path.join(data_dir, target_name,
                                f'det/coadd/{target_name}_coadd_det.fits'
                                )
    print(f'using detection catalog {detect_cat}')
    print(f'using detection image {detect_im}')

    try:
        assert os.path.exists(detect_im) is True
        hdr = fits.getheader(detect_im)
        xcen = hdr['CRPIX1']; ycen = hdr['CRPIX2']
        coadd_center = [xcen, ycen]
        print(f'Read image data and setting image NFW center to ({xcen},{ycen})')

    except Exception as e:
        print('\n\n\nNo coadd image center found, cannot calculate tangential shear\n\n.')
        raise e

    if nfw_seed is None:
        nfw_seed = utils.generate_seeds(1)

    ## n.b outfile is the name of the metacalibrated &
    ## quality-selected galaxy catalog

    cat_info={
        'detect_cat': detect_cat,
        'mcal_file': mcal_file,
        'run_name': target_name,
        'mcal_selected': outfile,
        'outdir': outdir,
        'truth_file': truth_file,
        'nfw_file': nfw_file,
        'Nresample': Nresample,
        'nfw_seed': nfw_seed
    }

    annular_info = {
        'rmin': rmin,
        'rmax': rmax,
        'nbins': nbins,
        'coadd_center': coadd_center,
        'xy_args': xy_cols,
        'shear_args': shear_args
    }

    annular_cat = AnnularCatalog(cat_info, annular_info)

    # run everything
    annular_cat.run(overwrite=overwrite, vb=vb)

    return 0

if __name__ == '__main__':

    args = parse_args()

    rc = main(args)

    if rc == 0:
        print('make_annular_catalog.py has completed succesfully')
    else:
        print(f'make_annular_catalog.py has failed w/ rc={rc}')
