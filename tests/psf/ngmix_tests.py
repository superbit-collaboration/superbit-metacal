import numpy as np
import os
from multiprocessing import Pool
import ngmix
import galsim
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import superbit_lensing.utils as utils
from espy import images
import pudb

parser = ArgumentParser()

parser.add_argument('outfile', type=str, help='Name of outfile')
parser.add_argument('--config', type=str, default=None,
                    help='Config filename to set parameters from')
parser.add_argument('--outdir', type=str, default=None, help='Name of outdir')
parser.add_argument('--run_name', type=str, default='', help='Name of run')
parser.add_argument('--ncores', type=int, default=1, help='Number of CPU cores to use')
parser.add_argument('--nobjs', type=int, default=1, help='Number of objects to simulate')
parser.add_argument('--nreal', type=int, default=1, help='Number of realizations per object')
parser.add_argument('--seed', type=int, default=72396, help='Number of CPU cores to use')
parser.add_argument('-v', '--verbose', action='store_true', default=False, help='Verbose')

def setup_test_config(gal_model='exp', psf_model='gauss'):

    config = {
        'gal_model': gal_model,
        'psf_model': psf_model,
        'pixel_scale': 0.144, # arcsec/pixel
        # 'psf_scale': 0.144, # arcsec/pixel
        'centroid_offset': 0, # pixels
        'shift': False,
        'nobs': 1,
        'gal_flux': 1000.,
        'gal_hlr': 2., # arcsec
        'psf_noise': 1e-3,
        'gal_noise': 1e-1,
        # 'psf_noise': 1e-9,
        # 'gal_noise': 1e-9,
        'psf_fwhm': 0.1, # arcsec
        'g1': 0.,
        'g2': 0.,
        }

    return config

def check_set_default(config, name, default):
    if not hasattr(config, name):
        config[name] = default

    return

def setup_priors(config):
    '''ideally this would all be checked by the config, but still
       a work in progress
    '''

    if 'T_range' in config:
        T_range = config['T_range']
    else:
        T_range = [-1.0, 1.e3]
        config['T_range'] = T_range

    if 'F_range' in config:
        F_range = config['F_range']
    else:
        F_range = [-100.0, 1.e9]
        config['F_range'] = F_range

    if 'g_sigma' in config:
        sigma = config['g_sigma']
    else:
        sigma = 0.1

    if 'nband' in config:
        nband = config['nband']
    else:
        nband = None

    rng = config['rng']
    scale = config['pixel_scale']

    g_prior = ngmix.priors.GPriorBA(sigma=sigma, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0, cen2=0, sigma1=scale, sigma2=scale, rng=rng,
    )

    T_prior = ngmix.priors.FlatPrior(minval=T_range[0], maxval=T_range[1], rng=rng)
    F_prior = ngmix.priors.FlatPrior(minval=F_range[0], maxval=F_range[1], rng=rng)

    if nband is not None:
        F_prior = [F_prior]*nband

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior

def make_model(config, obj, model):
    '''
    obj is gal or psf
    model is profile type
    '''

    if obj == 'psf':
        flux = 1
    else:
        flux = config[f'{obj}_flux']

    if model == 'gauss':
        try:
            sigma = config[f'{obj}_sigma']
        except KeyError:
            fwhm = config[f'{obj}_fwhm']
            sigma = utils.fwhm2sigma(fwhm)
        gs = galsim.Gaussian(flux=flux, sigma=sigma)

    elif model == 'exp':
        hlr = config[f'{obj}_hlr']
        gs = galsim.Exponential(half_light_radius=hlr, flux=flux)

    else:
        raise ValueError('Warning: `make_data` is not yet implemented ' +
                         f'for gal_model={gal_model}!')

    return gs

def make_obj(i, config):
    '''
    Right now, just making a specific object. But could do something
    more sophisticated depending on i
    '''

    rng = config['rng']
    scale = config['pixel_scale']
    psf_noise = config['psf_noise']
    fwhm = config['psf_fwhm']
    flux = config['gal_flux']
    g1, g2 = config['g1'], config['g2']

    # shift params
    if config['shift'] is True:
        dx, dy = rng.uniform(low=-scale/2., high=scale/2., size=2)
    else:
        dx, dy = 0., 0.

    # Make galaxy
    gal = make_model(config, 'gal', config['gal_model'])
    gal.shear(g1=g1, g2=g2)
    gal.shift(dx=dx, dy=dy)

    # Make PSF
    psf = make_model(config, 'psf', config['psf_model'])

    # Make object observation
    obj = galsim.Convolve(psf, gal)

    psf_im = psf.drawImage(scale=scale).array
    obj_im = obj.drawImage(scale=scale).array

    # Add noise
    obj_noise = config['gal_noise']
    psf_noise = config['psf_noise']
    psf_im += rng.normal(scale=psf_noise, size=psf_im.shape)
    obj_im += rng.normal(scale=obj_noise, size=obj_im.shape)

    obj_cen = (np.array(obj_im.shape)-1.0)/2.0
    psf_cen = (np.array(psf_im.shape)-1.0)/2.0

    # Setup Jacobians
    obj_jacobian = ngmix.DiagonalJacobian(
        row=obj_cen[0]+dy/scale, col=obj_cen[1]+dx/scale, scale=scale,
    )
    psf_jacobian = ngmix.DiagonalJacobian(
        row=psf_cen[0], col=psf_cen[1], scale=scale,
    )

    # Setup weights
    obj_wt = obj_im*0 + 1.0/obj_noise**2
    psf_wt = psf_im*0 + 1.0/psf_noise**2

    # Setup ngmix observations
    psf_obs = ngmix.Observation(
        psf_im,
        weight=psf_wt,
        jacobian=psf_jacobian
    )
    obj_obs = ngmix.Observation(
        obj_im,
        weight=obj_wt,
        jacobian=obj_jacobian,
        psf=psf_obs
    )

    return obj_obs

def make_data(config):
    '''
    Returns list of [(ngmix.Observation, pars dict)] for each source
    '''

    nobjs = config['nobjs']
    ncores = config['ncores']

    obs_list = ngmix.observation.ObsList()

    out = []

    with Pool(ncores) as pool:
        obs_list.append(pool.starmap(make_obj,
                                [(i, config) for i in range(nobjs)]
                                )[0]
                        )

    return obs_list

def main():
    '''
    This script tries to test how accurately we can capture
    object photometry with ngmix when there is no model mis-specification;
    i.e. we use the "true" models and priors for fitting

    What we do:
    - Choose PSF and galaxy model from a config
    - Setup fake observations using these models
    - - Do this for noisy and noiseless versions
    - Fit idealized observation images with ngmix, using the correct
      model profiles & priors
    - Do this for Nobjs
    - Do this for Nreal realizations

    Questions:
    - Should we do a "ring test", i.e. rotate each source at least once?
    - Should we instead draw from a distribution of galaxy properties?
    '''

    args = parser.parse_args()
    config_file = args.config
    outfile = args.outfile
    outdir = args.outdir
    run_name = args.run_name
    nobjs = args.nobjs
    nreal = args.nreal
    ncores = args.ncores
    seed = args.seed
    vb = args.verbose

    if outdir is not None:
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    logfile = f'{run_name}-ngmix-test.log'
    logdir = outdir
    log = utils.setup_logger(logfile, logdir=logdir)

    logprint = utils.LogPrint(log, vb)

    if config_file is None:
        logprint('Setting up test config')
        config = setup_test_config()
    else:
        logprint(f'Reading config file {config_file}')
        config = utils.read_yaml(config_file)

    for key, val in vars(args).items():
        config[key] = val

    logprint(f'Using seed = {seed}')
    rng = np.random.RandomState(seed)
    config['rng'] = rng

    logprint('Setting up priors')
    priors = setup_priors(config)

    logprint('Making data')
    obs_list = make_data(config)

    # Will eventually do a mcal setup:
    # Metacal setup
    # mcal_shear = 0.01
    # metacal_pars={'step':mcal_shear}
    # lm_pars = {'maxfev':2000, 'xtol':5.0e-5, 'ftol':5.0e-5}
    # max_pars = {'method':'lm','lm_pars':lm_pars,'find_center':True}
    # prior=_get_priors()
    # Tguess=4*jj_im.get_scale()**2
    # ntry=3

    for i in range(nobjs):
        logprint(f'Starting object {i}')

        obs = obs_list[i]
        boot = ngmix.Bootstrapper(obs)

        psf_Tguess = config['psf_fwhm']*config['pixel_scale']**2

        boot.fit_psfs(
            config['psf_model'],
            psf_Tguess,
        )

        pars = {
            'method': 'lm',
            'lm_pars': {},
        }

        logprint('Starting fit...')
        boot.fit_max(
            config['gal_model'],
            pars,
        )

        res = boot.get_fitter().get_result()
        assert res['flags'] == 0

        make_plots(config, obs, boot, logprint)

        # Will eventually use mcal:
        # mcal_boot = ngmix.bootstrap.MaxMetacalBootstrapper(obs_list)
        # mcal_boot.fit_metacal(config['psf_model'],
        #                       config['gal_model'],
        #                       max_pars,
        #                       Tguess,
        #                       prior=priors,
        #                       ntry=ntry,
        #                       metacal_pars=mcal_pars)

    logprint('Done!')

    return

def make_plots(config, obs, boot, logprint):

    run_name = config['run_name']
    outdir = config['outdir']

    ###############################################
    # Make PSF plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))

    psf_obs = obs.psf.image
    psf_jac = obs.psf.jacobian
    psf_gm = boot.psf_fitter.get_gmix()
    psf_model = psf_gm.make_image(psf_obs.shape, jacobian=psf_jac)

    im1 = ax1.imshow(psf_obs, origin='lower')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Observed PSF')

    im2 = ax2.imshow(psf_model, origin='lower')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Model PSF')

    im3 = ax3.imshow(psf_obs - psf_model, origin='lower')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('Observed - Model PSF')

    fwhm = config['psf_fwhm']
    fig.suptitle(f'Run {run_name}: Exponential Galaxy + Gaussian PSF; FWHM={fwhm}')

    outfile = f'ngmix-test-{run_name}-psf-compare-fwhm-{fwhm}.png'
    fig.savefig(os.path.join(outdir, outfile), bbox_inches='tight')
    plt.show()
    plt.close()

    ###############################################
    # Make obs plots
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(14, 5))

    gal_obs = obs.image
    gal_jac = obs.jacobian
    gal_gm = boot.get_fitter().get_convolved_gmix()
    gal_model = gal_gm.make_image(gal_obs.shape, jacobian=gal_jac)

    im1 = ax1.imshow(gal_obs, origin='lower')
    plt.colorbar(im1, ax=ax1)
    ax1.set_title('Observed source')

    im2 = ax2.imshow(gal_model, origin='lower')
    plt.colorbar(im2, ax=ax2)
    ax2.set_title('Model profile')

    im3 = ax3.imshow(gal_obs - gal_model, origin='lower')
    plt.colorbar(im3, ax=ax3)
    ax3.set_title('Observed - Model source')

    fig.suptitle(f'Run {run_name}: Exponential Galaxy + Gaussian PSF; FWHM={fwhm}')

    outfile = f'ngmix-test-{run_name}-gal-compare-fwhm-{fwhm}.png'
    fig.savefig(os.path.join(outdir, outfile), bbox_inches='tight')
    plt.show()
    plt.close()

    return

if __name__ == '__main__':
    main()
