import ngmix
from ngmix.medsreaders import NGMixMEDS
from ngmix.fitting import Fitter
import numpy as np
import os
import time
from astropy.table import Table, vstack, hstack
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('medsfile', type=str,
                    help='MEDS file to process')
parser.add_argument('outfile', type=str,
                    help='Output filename')
parser.add_argument('-outdir', type=str, default=None,
                    help='Output directory')
parser.add_argument('-start', type=int, default=None,
                    help='Starting index for MEDS processing')
parser.add_argument('-end', type=int, default=None,
                    help='Ending index for MEDS processing')
parser.add_argument('-seed', type=int, default=None,
                    help='Seed for initializing rng')
parser.add_argument('-gal_model', type=str, default='gauss',
                    help='ngmix model for galaxy profile')
parser.add_argument('-psf_model', type=str, default='gauss',
                    help='ngmix model for star profile')
parser.add_argument('-shear_step', type=float, default=0.01,
                    help='Mcal shear applied during finite difference')
parser.add_argument('--clobber', action='store_true', default=False,
                    help='Set to overwrite output files')
parser.add_argument('--vb', action='store_true', default=False,
                    help='Make verbose')

class MetacalRunner(object):
    def __init__(self, medsfile, vb=False):
        self.medsfile = medsfile
        self.vb = vb

        self.meds = NGMixMEDS(medsfile)
        self.cat = self.meds.get_cat()
        self.Nobjs = len(self.cat)

        self.pixel_scale = self.get_pixel_scale(0)

        self.prior = None
        self.guesser = None

        self.gal_model = None
        self.psf_model = None
        self.shear_step = None
        self.seed = None
        self.rng = None

        return

    def set_seed(self, seed=None):
        if seed is None:
            # quickly check if it has been set explicitly
            try:
                seed = self.seed
                return
            except AttributeError:
                # current time in microseconds
                seed = int(1e6*time.time())

        self.seed = seed

        return

    def set_rng(self):
        self.rng = np.random.RandomState(self.seed)

        return

    def get_pixel_scale(self, iobj):
        '''
        Get pixel scale from image jacobian
        NOTE: Assumes same pix_scale for all cutouts
        '''

        jac_list = self.get_jacobians(iobj)
        return jac_list[iobj].get_scale()

    def _setup_guessers(self):
        '''
        Sets up guesser given a pixel scale. Prior must be setup first
        '''

        if self.prior is None:
            raise ValueError('prior must be setup before guesser!')

        Tguess = 4*self.pixel_scale**2

        # make parameter guesses based on a psf flux and a rough T
        self.guesser = ngmix.guessers.TPSFFluxAndPriorGuesser(
            rng=self.rng,
            T=Tguess,
            prior=self.prior,
        )

        return

    def setup_bootstrapper(self, gal_model, psf_model, shear_step):
        '''
        Initialize ngmix bootstrapper for metacal measurement
        '''

        self.gal_model = gal_model
        self.psf_model = psf_model
        self.shear_step = shear_step

        #----------------------------------------------------------------------
        # Setup prior & guesser
        self._setup_prior()
        self._setup_guessers()

        #----------------------------------------------------------------------
        # Setup fitters

        # Fairly standard par values for lm fit
        lm_pars = {
                'maxfev': 2000,
                'ftol': 1.0e-05,
                'xtol': 1.0e-05,
            }

        psf_fitter = Fitter(
            model=self.psf_model, fit_pars=lm_pars
            )
        fitter = Fitter(
            model=self.gal_model, prior=self.prior, fit_pars=lm_pars
            )

        # from ngmix mcal example:
        # We will measure moments with a fixed gaussian weight function
        # weight_fwhm = 1.2
        # psf_fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)
        # fitter = ngmix.gaussmom.GaussMom(fwhm=weight_fwhm)

        # the runners run the measurement code on observations
        psf_runner = ngmix.runners.PSFRunner(
            fitter=psf_fitter, #guesser=self.psf_guesser
            )
        runner = ngmix.runners.Runner(
            fitter=fitter, guesser=self.guesser
            )

        #----------------------------------------------------------------------
        # the bootstrapper automates the metacal image shearing as well as both psf
        # and object measurements
        self.boot = ngmix.metacal.MetacalBootstrapper(
            runner=runner,
            psf_runner=psf_runner,
            step=shear_step,
            rng=self.rng,
            # TODO: check into this later, we may want to specify PSF explicitly
            # psf=args.psf
        )

        return

    def go(self, start, end):
        '''
        Run the metacal measurement on all observations from
        `start` to `end`
        '''

        if end < start:
            raise ValueError('end must be greater than start!')
        N = end - start

        mcal_tabs = []
        for iobj in range(start, end):
            if self.vb is True:
                if iobj // 10 == 0:
                    print(f'Starting {iobj} of {N}...')
            obs = self.get_obslist(iobj)
            obj_info = self.get_obj_info(iobj)

            res_dict, obs_dict = self.boot.go(obs)

            res_dict = self.add_mcal_responsivities(res_dict)
            res_tab = self.mcal_dict2tab(res_dict, obj_info)
            mcal_tabs.append(res_tab)

        mcal_tabs = vstack(mcal_tabs)

        return mcal_tabs

    def get_obslist(self, iobj):
        return self.meds.get_obslist(iobj)

    def get_jacobians(self, iobj):
        Njac = len(self.meds.get_jacobian_list(iobj))

        jacobians = [self.meds.get_ngmix_jacobian(iobj, icutout)
                     for icutout in range(Njac)]

        return jacobians

    def get_obj_info(self, iobj):
        '''
        Setup object property dictionary used to compile fit params later on
        '''

        obj = self.meds[iobj]

        # Mcal object properties
        obj_info = {}

        obj_info['meds_indx'] = iobj
        obj_info['id'] = obj['id']
        obj_info['ra'] = obj['ra']
        obj_info['dec'] = obj['dec']
        obj_info['X_IMAGE'] = obj['X_IMAGE']
        obj_info['Y_IMAGE'] = obj['Y_IMAGE']

        return obj_info

    def _setup_prior(self):

        pixel_scale = self.pixel_scale

        rng = self.rng

        # prior on ellipticity.  The details don't matter, as long
        # as it regularizes the fit.  This one is from Bernstein & Armstrong 2014

        g_sigma = 0.3
        g_prior = ngmix.priors.GPriorBA(g_sigma, rng=rng)

        # 2-d gaussian prior on the center
        # row and column center (relative to the center of the jacobian, which would be zero)
        # and the sigma of the gaussians

        # units same as jacobian, probably arcsec
        row, col = 0.0, 0.0
        row_sigma, col_sigma = pixel_scale, pixel_scale # use pixel_scale as a guess
        cen_prior = ngmix.priors.CenPrior(row, col, row_sigma, col_sigma, rng=rng)

        # T prior.  This one is flat, but another uninformative you might
        # try is the two-sided error function (TwoSidedErf).
        # NOTE: T units are arcsec^2 but can be slightly negative, especially for
        # stars if the PSF is mis-estimated

        Tminval = -1.0 # arcsec squared
        Tmaxval = 1000
        T_prior = ngmix.priors.FlatPrior(Tminval, Tmaxval, rng=rng)

        # similar for flux.  Make sure the bounds make sense for
        # your images

        Fminval = -1.e1
        Fmaxval = 1.e9
        F_prior = ngmix.priors.FlatPrior(Fminval, Fmaxval, rng=rng)

        # now make a joint prior.  This one takes priors
        # for each parameter separately
        self.prior = ngmix.joint_prior.PriorSimpleSep(
            cen_prior,
            g_prior,
            T_prior,
            F_prior,
            )

        return

    def add_mcal_responsivities(self, mcal_res):
        '''
        Compute and add the mcal responsivity values to the output
        result dict from get_metacal_result()

        NOTE: These are only for the selection-independent component!
        '''

        mcal_shear = self.shear_step

        # Define full responsivity matrix, take inner product with shear moments
        r11 = (mcal_res['1p']['g'][0] - mcal_res['1m']['g'][0]) / (2*mcal_shear)
        r12 = (mcal_res['2p']['g'][0] - mcal_res['2m']['g'][0]) / (2*mcal_shear)
        r21 = (mcal_res['1p']['g'][1] - mcal_res['1m']['g'][1]) / (2*mcal_shear)
        r22 = (mcal_res['2p']['g'][1] - mcal_res['2m']['g'][1]) / (2*mcal_shear)

        R = [ [r11, r12], [r21, r22] ]
        Rinv = np.linalg.inv(R)
        gMC = np.dot(Rinv,
                     mcal_res['noshear']['g']
                     )

        MC = {
            'r11':r11, 'r12':r12,
            'r21':r21, 'r22':r22,
            'g1_MC':gMC[0], 'g2_MC':gMC[1]
        }

        mcal_res['MC'] = MC

        return mcal_res

    def mcal_dict2tab(self, mcal, obj_info):
        '''
        mcal is the dict returned by ngmix.get_metacal_result()

        obj_info is an array with MEDS identification info like id, ra, dec
        not returned by the function
        '''

        # Annoying, but have to do this to make Table from scalars
        for key, val in obj_info.items():
            obj_info[key] = np.array([val])

        tab_names = ['noshear', '1p', '1m', '2p', '2m','MC']
        for name in tab_names:
            tab = mcal[name]

            for key, val in tab.items():
                tab[key] = np.array([val])

            mcal[name] = tab

        id_tab = Table(data=obj_info)

        tab_noshear = Table(mcal['noshear'])
        tab_1p = Table(mcal['1p'])
        tab_1m = Table(mcal['1m'])
        tab_2p = Table(mcal['2p'])
        tab_2m = Table(mcal['2m'])
        tab_MC = Table(mcal['MC'])

        join_tab = hstack([id_tab, hstack([tab_noshear,
                                           tab_1p,
                                           tab_1m,
                                           tab_2p,
                                           tab_2m,
                                           tab_MC
                                           ],
                                          table_names=tab_names)
                           ]
                          )

        return join_tab

    def fit_obj(self, iobj, pars=None, ntry=4, vb=False):
        '''
        Run metacal fit for a single object of given index

        pars: mcal running parameters
        '''

        obj_info = self.get_obj_info(iobj)

        # Fits need a list of ngmix.Observation objects
        obs_list = self.get_obs_list(iobj)

        # Get pixel scale from image jacobian
        jac_list = self.get_jacobians(iobj)
        pixel_scale = jac_list[0].get_scale()

        if pars is None:
            # standard mcal run parameters
            mcal_shear = 0.01
            lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
            max_pars = {'method':'lm', 'lm_pars':lm_pars, 'find_center':True}
            metacal_pars = {'step':mcal_shear}
        else:
            mcal_shear = metacal_pars['step']
            max_pars = pars['max_pars']
            metacal_pars = pars['metacal_pars']

        prior = self.get_prior(pixel_scale)

        Tguess = 4*pixel_scale**2

        # setup run bootstrapper
        mcb = ngmix.bootstrap.MaxMetacalBootstrapper(obs_list)

        # Run the actual metacalibration step on the observed source
        mcb.fit_metacal(self.psf_model, self.gal_model, max_pars, Tguess, prior=prior,
                        ntry=ntry, metacal_pars=metacal_pars)

        mcal_res = mcb.get_metacal_result() # this is a dict

        # Add selection-independent responsitivities
        mcal_res = self.add_mcal_responsivities(mcal_res)

        if vb is True:
            r11 = mcal_res['MC']['r11']
            r22 = mcal_res['MC']['r22']
            print(f'R11: {r11:.3} \nR22: {r22:.3} ')

        mcal_tab = self.mcal_dict2tab(mcal_res, obj_info)

        return mcal_tab

def main(args):

    medsfile = args.medsfile
    outfile = args.outfile
    outdir = args.outdir
    start = args.start
    end = args.end
    seed = args.seed
    gal_model = args.gal_model
    psf_model = args.psf_model
    shear_step = args.shear_step
    clobber = args.clobber
    vb = args.vb # if True, prints out values of R11/R22 for every galaxy

    if outdir is not None:
        outfile = os.path.join(outdir, outfile)

    mcal_runner = MetacalRunner(medsfile, vb=vb)

    mcal_runner.set_seed(seed)
    mcal_runner.set_rng()

    mcal_runner.setup_bootstrapper(
        gal_model=gal_model, psf_model=psf_model, shear_step=shear_step
        )

    if start is None:
        start = 0
    if end is None:
        end = mcal_runner.Nobjs

    Tstart = time.time()

    mcal_tabs = mcal_runner.go(start, end)

    Tend = time.time()

    T = Tend - Tstart
    print(f'Total fitting and stacking time: {T} seconds')

    if vb is True:
        print(f'Writing out mcal results to {outfile}...')
    mcal_tabs.write(outfile, overwrite=clobber)

    if vb is True:
        print('Done!')

    return

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
