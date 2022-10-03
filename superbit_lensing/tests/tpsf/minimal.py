import numpy as np
import ngmix
import meds
import galsim as gs
from astropy.table import Table, vstack, join
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool
from argparse import ArgumentParser

from prior import GaussianPrior, UniformPrior
from superbit_lensing import utils
import superbit_lensing.metacalibration.mcal_runner as _mcal

import ipdb

'''
This test script builds a minimal MEDS file from simulated sources
with perfect PSF information to determine the root cause of the
metacal Tpsf bias seen in our main pipeline
'''

def parse_args():

    parser = ArgumentParser()

    parser.add_argument('Nobjs', type=int,
                        help='The number of galaxies to simulate')
    parser.add_argument('-ncores', type=int, default=8,
                        help='The number of CPU cores to use')
    parser.add_argument('-fwhm', type=float, default=0.48,
                        help='The PSF FWHM in arcsec')
    parser.add_argument('-sky_sigma', type=float, default=1e-1,
                        help='The background sky noise')
    parser.add_argument('-psf_noise', type=float, default=1e-6,
                        help='Small amounts of PSF noise')
    parser.add_argument('-im_pix_scale', type=float, default=0.141,
                        help='The image pixel scale in arcsec/pixel')
    parser.add_argument('-psf_pix_scale', type=float, default=0.141,
                        help='The psf pixel scale in arcsec/pixel')
    parser.add_argument('-im_stamp_size', type=int, default=48,
                        help='The stamp size for the image stamp')
    parser.add_argument('-psf_stamp_size', type=int, default=17,
                        help='The stamp size for the PSF stamp')
    parser.add_argument('-seed', type=int, default=None,
                        help='The seed to use')
    parser.add_argument('--plot_all', action='store_true', default=False,
                        help='Turn on to make per-object plots')
    parser.add_argument('--show', action='store_true', default=False,
                        help='Turn on to show plots')

    return parser.parse_args()

class TpsfTestRunner(object):

    def __init__(self, args):
        '''
        args are the return of parse_args()
        '''

        for key, val in vars(args).items():
            setattr(self, key, val)

        self.set_seed()
        self._setup_source_prior()

        self.mcal_shear = 0.01

        return

    def set_seed(self):
        '''
        seed: int
            The random seed to use for mcal runs. Will set
            a default using time() if none is passed
        '''

        if self.seed is None:
            seed = np.random.randint(0, 2**32-1)

        self.seed = seed

        self._set_rng(seed)
        self._set_gs_rng(seed)

        return

    def _set_rng(self, seed):
        self.rng = np.random.RandomState(seed)

        return

    def _set_gs_rng(self, seed):
        self.gs_rng = gs.BaseDeviate(seed)

        return

    def _setup_source_prior(self):
        self.prior = {
            'n': UniformPrior(1, 3.5),
            'flux': GaussianPrior(mu=1e4, sigma=2.5e3),
            'inclination': UniformPrior(0, 0.9*np.pi/2.),
            'half_light_radius': GaussianPrior(mu=2., sigma=0.5),
        }

        return

    @staticmethod
    def _render_obj(obj, stamp_size, pix_scale, method='auto', noise=None):

        image = gs.Image(stamp_size, stamp_size, scale=pix_scale)

        obj.drawImage(image, method=method)

        if noise is not None:
            image.addNoise(noise)

        return image.array

    property
    def gs_im_noise(self):
        if not hasattr(self, '_gs_im_noise'):
            self._gs_im_noise = gs.GaussianNoise(
                self.gs_rng, self.sky_sigma
                )

        return self._gs_im_noise

    property
    def gs_psf_noise(self):
        if not hasattr(self, '_gs_psf_noise'):
            self._gs_psf_noise = gs.GaussianNoise(
                self.gs_rng, self.psf_noise
                )

        return self._gs_psf_noise

    property
    def psf(self):
        if not hasattr(self, '_psf'):
            self._psf = gs.Gaussian(flux=1, fwhm=self.fwhm)

        return self._psf

    def build_psf_cutouts(self):
        psf_list = []

        for i in range(self.Nobjs):
            psf_list.append(
                self._render_psf(self.psf())
                )

        return psf_list

    def _render_psf(self, psf, method='real_space'):
        '''
        We default to real_space rendering as drawing the PSF alone
        in k-space leads to unwanted artifacts
        '''

        psf_im = self._render_obj(
            psf, self.psf_stamp_size, self.psf_pix_scale, method=method,
            noise=self.gs_psf_noise()
            )

        return psf_im

    def build_source_cutouts(self):
        '''
        NOTE: This is the list of source images, not
        source objects
        '''

        source_list = []

        for i in range(self.Nobjs):
            source = self.generate_source()

            source_list.append(
                self._render_source(source)
                )

        return source_list

    def generate_source(self):
        '''
        Generate an inclined sersic galaxy using a simple
        prior on source parameters
        '''

        n = self.prior['n'].sample()
        inclination = self.prior['inclination'].sample()
        flux = self.prior['flux'].sample()
        hlr = self.prior['half_light_radius'].sample()

        inclination *= gs.radians

        source = gs.InclinedSersic(
            n, inclination, half_light_radius=hlr, flux=flux
            )

        # rotate source
        theta = np.random.uniform(0, np.pi) * gs.radians
        source = source.rotate(theta)

        return source

    def _render_source(self, source, method='auto'):

        conv_source = gs.Convolve(
            [self.psf(), source]
            )

        source_im = self._render_obj(
            conv_source, self.im_stamp_size, self.im_pix_scale, method=method,
            noise=self.gs_im_noise()
            )

        return source_im

    # NOTE: No longer building a MEDS file...
    def build_image_info(self, max_filepath_len=200):
        image_info = meds.util.get_image_info_struct(
            self.Nobjs, max_filepath_len
            )

        return image_info

    # NOTE: No longer building a MEDS file...
    def build_obs_list(self, im_list, psf_list):

        assert len(im_list) == self.Nobjs
        assert len(psf_list) == self.Nobjs

        obs_list = []

        for i in range(self.Nobjs):
            obs_list.append(
                self.build_obs(im_list[i], psf_list[i])
                )

        return obs_list

    def build_obs(self, im, psf_im):
        cen = (np.array(im.shape)-1.0)/2.0
        psf_cen = (np.array(psf_im.shape)-1.0)/2.0

        jacobian = ngmix.DiagonalJacobian(
            row=cen[0], col=cen[1], scale=self.im_pix_scale,
        )
        psf_jacobian = ngmix.DiagonalJacobian(
            row=psf_cen[0], col=psf_cen[1], scale=self.psf_pix_scale,
        )

        wt = im*0. + 1./self.sky_sigma**2
        psf_wt = psf_im*0 + 1./self.psf_noise**2

        psf_obs = ngmix.Observation(
            psf_im,
            weight=psf_wt,
            jacobian=psf_jacobian,
        )

        obs = ngmix.Observation(
            im,
            weight=wt,
            jacobian=jacobian,
            psf=psf_obs,
        )

        return obs

    def _setup_bootstrapper(self):

        pixel_scale = self.im_pix_scale

        # abuse our existing code
        class NoMEDSMetacalRunner(_mcal.MetacalRunner):
            def __init__(self, vb=False, logprint=None):
                '''
                Make MetacalRunner work w/o a MEDS file
                '''

                if logprint is None:
                    # don't log, just print
                    logprint = utils.LogPrint(None, vb)
                else:
                    vb = logprint.vb
                self.logprint = logprint
                self.vb = vb

                self.pixel_scale = pixel_scale

                self.fitter = None
                self.psf_fitter = None
                self.shear_step = None
                self.seed = None
                self.rng = None

                self.prior = None
                self.guesser = None
                self.psf_guesser = None
                self.lm_pars = None

                self.mcal_table = None

                return

        mcal_runner = NoMEDSMetacalRunner(vb=True)
        mcal_runner.set_seed(self.seed)

        gal_fitter = 'fitter'
        gal_kwargs = {'model': 'gauss'}
        psf_fitter = 'fitter'
        psf_kwargs = {'model': 'gauss'}
        mcal_runner.setup_bootstrapper(
            gal_fitter, psf_fitter, 0.01,
            gal_kwargs=gal_kwargs, psf_kwargs=psf_kwargs,
            ntry=3
            )

        self.boot = mcal_runner.boot

        return

    def run_mcal(self, obs_list):

        self._setup_bootstrapper()

        print(f'Running on {self.ncores} cores')
        if self.ncores > 1:
            with Pool(self.ncores) as pool:
                mcal_table = vstack(pool.starmap(
                    self._fit_one,
                    [(i,
                    self.boot,
                    obs_list[i],
                    Table(),
                    self.mcal_shear
                    ) for i in range(0, self.Nobjs)
                    ])
                )
        else:
            mcal_table = vstack(
                [self._fit_one(
                    i, self.boot, obs_list[i], Table(), self.mcal_shear
                    ) for i in range(0, self.Nobjs)
                 ]
                )

        Nfailed = self.Nobjs - len(obs_list)
        print(f'{Nfailed} objects failed metacalibration fitting ' +\
                      'and are excluded from output catalog')
        print('Done!')

        return mcal_table

    @staticmethod
    def _fit_one(iobj, bootstrapper, obs, obj_info, mcal_shear):
        '''
        A static method to wrap the mcal fitting to allow
        for multiprocessing

        iobj: int
            The MEDS index of the object to fit
        bootstrapper: ngmix.bootstrapper
            The ngmix bootstrapper that will automate the mcal fit
        obs: ngmix.Observation
            All requisite images & metadata of the obj to fit
        mcal_shear: float
            The applied shear to the metacal images
        '''

        print(f'Starting fit for obj {iobj}')

        try:
            res_dict, obs_dict = bootstrapper.go(obs)

            # compute value-added cols such as gamma-only responsivity,
            # PSF size, "roundified" s2n, etc.
            _mcal.add_mcal_cols(res_dict, obs_dict, mcal_shear)

            # ipdb.set_trace()

            return _mcal.mcal_dict2tab(res_dict, obs_dict, obj_info)

        except Exception as e:
            print(f'object {iobj}: Exception: {e}')
            print(f'object {iobj} failed, skipping...')

            return Table()

    def make_plots(self, source_list, psf_list, obs_list, mcal,
                   outdir=None):

        if self.plot_all is True:
            # Make per-object cutout plots
            size = (10,4)
            for i, obs in enumerate(obs_list):
                im = source_list[i]
                psf = psf_list[i]

                plt.subplot(121)
                plt.imshow(im)
                plt.colorbar()
                plt.title('source image')

                plt.subplot(122)
                plt.imshow(psf)
                plt.colorbar()
                fwhm = self.fwhm
                meas_Tpsf = mcal['Tpsf_noshear'][i]
                meas_fwhm = ngmix.moments.T_to_fwhm(meas_Tpsf)
                plt.title(f'source psf\nFWHM={fwhm}; meas_fwhm={meas_fwhm:.4f}')

                plt.gcf().set_size_inches(size)

                outfile = os.path.join(outdir, f'cutout_{i}.png')
                plt.savefig(outfile, bbox_inches='tight', dpi=300)
                plt.close()

        # make Tpsf comparison plot
        plt.subplot(121)
        plt.hist(mcal['Tpsf_noshear'], ec='k', bins=20)
        plt.xlabel('Tpsf_noshear')
        plt.title('Fitted PSF Tpsf')

        plt.subplot(122)
        fwhm = ngmix.moments.T_to_fwhm(mcal['Tpsf_noshear'])
        plt.hist(fwhm, ec='k', bins=20)
        plt.xlabel('ngmix.moments.T_to_fwhm(Tpsf_noshear)')
        ref_fwhm = 0.48
        plt.axvline(ref_fwhm, ls='--', c='k', lw=2, label='True PSF FWHM')
        plt.legend()
        plt.title('Fitted PSF fwhm')

        plt.gcf().set_size_inches(12,5)

        outfile = os.path.join(outdir, f'Tpsf_compare.png')
        plt.savefig(outfile, bbox_inches='tight', dpi=300)

        if self.show is True:
            plt.show()
        else:
            plt.close()

        return

    def go(self):

        outdir = os.path.join(
            utils.MODULE_DIR, 'tests', 'tpsf', 'out'
            )
        plotdir = os.path.join(
            outdir, 'plots'
            )

        utils.make_dir(outdir)
        utils.make_dir(plotdir)

        # generate PSF cutouts
        psf_list = self.build_psf_cutouts()

        # genereate sources
        source_list = self.build_source_cutouts()

        obs_list = self.build_obs_list(source_list, psf_list)

        # NOTE: No longer building a MEDS file...
        # setup image info
        # image_info = self.build_image_info()

        # build MEDS file
        # meds = meds.maker.MEDSMaker(
        #     obj_info, image_info, config=meds_config,
        #     psf_data=psf_list, meta_data=meta
        #     )

        mcal_res = self.run_mcal(obs_list)

        self.make_plots(
            source_list, psf_list, obs_list, mcal_res,
            outdir=plotdir
            )

        return

def main(args):

    runner = TpsfTestRunner(args)

    runner.go()

    return

if __name__ == '__main__':
    args = parse_args()
    main(args)
