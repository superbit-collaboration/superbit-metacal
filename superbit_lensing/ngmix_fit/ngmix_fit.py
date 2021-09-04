import numpy as np
import os
import ngmix
from ngmix.medsreaders import NGMixMEDS
from multiprocessing import Pool
from argparse import ArgumentParser
import superbit_lensing.utils as utils
import pudb

parser = ArgumentParser()

parser.add_argument('--meds_file', type=str,
                    help='MEDS filename to run ngmix on')
parser.add_argument('--config', type=str,
                    help='ngmix config filename')
parser.add_argument('--outdir', type=str, default=None,
                    help='Output directory')
parser.add_argument('--n', type=int, default=1,
                    help='Number of CPU cores to use')
parser.add_argument('--test', action='store_true', default=False,
                    help='Use to do a test run w/ provided config')
parser.add_argument('-v', action='store_true', default=False,
                    help='Verbose')

class SuperBITngmixRunner(object):
    _req_fields = ['gal', 'psf', 'priors', 'fit_pars',
                   'run_name', 'pixel_scale', 'nband',
                   'seed']
    _opt_fields = []

    _gal_req_fields = ['model']
    _psf_req_fields = ['model']
    _priors_req_fields = ['T_range',
                          'F_range',
                          'g_sigma']

    def __init__(self, meds_file, config, logprint):

        assert isinstance(meds_file, str)
        self.meds_file = meds_file

        if isinstance(config, str):
            config = utils.read_yaml(config)
        else:
            assert isinstance(config, dict)

        utils.check_fields(config,
                           self._req_fields,
                           self._opt_fields,
                           'ngmix')

        field_zip = zip(['gal', 'psf', 'priors'],
                        [self._gal_req_fields,
                         self._psf_req_fields,
                         self._priors_req_fields]
                        )
        for field, req in field_zip:
            field_config = config[field]
            utils.check_req_fields(field_config,
                                   req,
                                   f'ngmix `{field}`'
                                   )

        self.config = config

        assert isinstance(logprint, utils.LogPrint)
        self.logprint = logprint

        seed = config['seed']
        rng = np.random.RandomState(seed)
        self.set_rng(rng)

        self._setup_priors()
        self._setup_meds()

        return

    def set_rng(self, rng):
        self.rng = rng

    def _setup_priors(self):
        self.logprint('Setting up ngmix priors')

        gal_model = self.config['gal']['model']
        imscale = self.config['pixel_scale']
        rng = self.rng

        # do common prior setup
        g_sigma = self.config['priors']['g_sigma']
        g_prior = ngmix.priors.GPriorBA(sigma=g_sigma, rng=rng)

        cen_prior = ngmix.priors.CenPrior(
            cen1=0, cen2=0, sigma1=imscale, sigma2=imscale, rng=rng
            )

        T_range = self.config['priors']['T_range']
        T_prior = ngmix.priors.FlatPrior(minval=T_range[0],
                                            maxval=T_range[1],
                                            rng=rng)

        F_range = self.config['priors']['F_range']
        F_prior = ngmix.priors.FlatPrior(minval=F_range[0],
                                            maxval=F_range[1],
                                            rng=rng)

        if self.config['nband'] > 1:
            F_prior = [F_prior]*nband

        cols = ['cen', 'g', 'T', 'F']
        prior_list = [cen_prior, g_prior, T_prior, F_prior]

        if gal_model == 'bdf':
            # setup fracdev
            fd_mean = self.config['priors']['fracdev_mean']
            fd_sig = self.config['priors']['fracdev_sigma']
            fracdev_prior = ngmix.priors.Normal(mean=fd_mean,
                                                sigma=fd_sig,
                                                rng=rng)

            cols.append('fracdev')
            prior_list.append(fracdev_prior)

        else:
            raise ValueError(f'{gal_model} prior not yet implemented!')

        prior_dict = {}
        for col, prior in zip(cols, prior_list):
            prior_dict[col] = prior

        priors = build_ngmix_priors(gal_model, prior_dict)

        self.priors = priors

        return

    def _setup_meds(self):

        self.logprint('Loading MEDS file')

        self.meds = NGMixMEDS(self.meds_file)
        self.nobjs = len(self.meds._cat)

        return

    def _setup_obs_list(self, i):
        '''i: MEDS index'''
        return self.meds.get_obslist(i)

    def fit(self, ncores):

        self.logprint('Starting ngmix fit')

        ngmix_fits = []
        with Pool(ncores) as pool:
            ngmix_fits.append(
                pool.starmap(_fit_one,
                             [(i,
                               self._setup_obs_list(i),
                               self.config,
                               self.logprint
                               ) for i in range(self.nobjs)]
                             )
                )

        return

def setup_bootstrapper(obs_list):
    boot = ngmix.Bootstrapper(obs_list)
    return boot

def _fit_one(i, obs_list, config, logprint):
    '''
    i: MEDS index
    obs_list: ngmix observation list for index i
    '''

    pars = config['fit_pars']

    boot = setup_bootstrapper(obs_list)

    logprint('Starting PSF fit')
    psf_Tguess = None
    boot.fit_psfs(
        config['psf']['model'],
        psf_Tguess
        )

    model = config['gal']['model']
    logprint(f'Starting {model} fit')
    boot.fit_max(
        model,
        pars
        )

    return boot

def build_ngmix_priors(model, prior_dict):
    '''
    model: ngmix galaxy model name
    prior_dict: dictionary of ngmix prior objects for each
                relevant parameter prior
    '''

    if model == 'bdf':
        priors = ngmix.joint_prior.PriorBDFSep(
            cen_prior=prior_dict['cen'],
            g_prior=prior_dict['g'],
            T_prior=prior_dict['T'],
            fracdev_prior=prior_dict['fracdev'],
            F_prior=prior_dict['F']
            )
    else:
        raise ValueError(f'{model} priors is not yet implemented!')

    return priors

def _make_test_ngmix_config():
    test_config = {
        'gal': {
            'model': 'bdf',
        },
        'psf': {
            'model': 'gauss'
        },
        'priors': {
            'T_range': [-1., 1.e3],
            'F_range': [-100., 1.e9],
            'g_sigma': 0.1,
            'fracdev_mean': 0.5,
            'fracdev_sigma': 0.1
        },
        'fit_pars': {
            'method': 'lm',
            'lm_pars': {}
        },
        'pixel_scale': 0.144, # arcsec / pixel
        'nband': 1,
        'seed': 172396,
        'run_name': 'bfd_test'
    }

    return test_config

def _return_test_medsfile():
    medsfile = os.path.join(utils.get_test_dir(),
                            'ngmix_fit',
                            'test_meds.fits')
    return medsfile

def main():
    args = parser.parse_args()
    meds_file = args.meds_file
    config_file = args.config
    outdir = args.outdir
    ncores = args.n
    test = args.test
    vb = args.v

    if test is True:
        config = _make_test_ngmix_config()
        meds_file = _return_test_medsfile()

        logfile = 'ngmix-fit-test.log'
        outdir = os.path.join(utils.TEST_DIR,
                              'ngmix_fit')
        log = utils.setup_logger(logfile, logdir=outdir)
        logprint = utils.LogPrint(log, vb)

    else:
        config = utils.read_yaml(config_file)
        assert meds_file is not None

        try:
            run_name = config['run_name']
        except KeyError:
            print('run_name is a required field in the ngmix config!')

        logfile = f'{run_name}-ngmix-fit.log'
        outdir = os.path.join(utils.TEST_DIR, 'ngmix_fit')
        log = utils.setup_logger(logfile, outdir=outdir)
        logprint = utils.LogPrint(log, vb)

    sb_runner = SuperBITngmixRunner(meds_file, config, logprint)

    rc = sb_runner.fit(ncores=ncores)

    print('Done!')

    return

if __name__ == '__main__':
    main()
