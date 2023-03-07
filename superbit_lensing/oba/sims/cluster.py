import numpy as np
import galsim
import math
from astropy.table import Table, Row, join
import astropy.units as u
from numpy.lib import recfunctions as rf
import ngmix

import photometry as phot
import instrument as inst
from superbit_lensing import utils

import ipdb

def make_a_cluster_galaxy(indx, obj, band, wcs, psf, camera, exp_time,
                          pix_scale, gs_params, logprint,
                          ra_col='ra_sim', dec_col='dec_sim', z_col='Z_LAMBDA',
                          flux_col_base='flux_adu'):
    '''
    Make a cluster galaxy given an input redmapper cluster member galaxy
    '''

    # these are the positions relative to the redmapper cluster center,
    # shifted to the near target center
    gal_ra, gal_dec = obj[ra_col], obj[dec_col]

    # actual ADU counts estimated for this galaxy in this SB band and at
    # the given exposure time
    flux_adu = obj[f'{flux_col_base}_{band}_{exp_time}']

    # grab morphological params matched to redmapper members from DDES Y3 GOLD
    g1, g2 = obj['SOF_CM_G_1'], obj['SOF_CM_G_2']
    T = obj['SOF_CM_T']
    fracdev = obj['SOF_CM_FRACDEV']
    TdByTe = obj['SOF_CM_TDBYTE']

    # make a ngmix GMix CM model (what was used in the fitting)
    gm_pars = [0.0, 0.0, g1, g2, T, flux_adu]
    gm = ngmix.gmix.GMixCM(fracdev, TdByTe, gm_pars)

    gal = gm.make_galsim_object(gsparams=gs_params)

    gal_ra *= galsim.degrees
    gal_dec *= galsim.degrees

    world_pos = galsim.CelestialCoord(gal_ra, gal_dec)
    image_pos = wcs.toImage(world_pos)

    # galaxy 2d position on sci img in arcsec
    gal_pos_cent_xasec = image_pos.x * pix_scale
    gal_pos_cent_yasec = image_pos.y * pix_scale

    gal_z = obj[z_col]

    # Position fractional stuff
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5

    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))

    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal

    offset = galsim.PositionD(dx, dy)

    # convolve galaxy GSObject with the psf (optical+jitter convolved already)
    convolution = galsim.Convolve(
        [psf, gal], gsparams=gs_params
        )

    gal_image = convolution.drawImage(
        wcs=wcs.local(image_pos),
        offset=offset,
        method='auto',
        )

    gal_image.setCenter(ix_nominal, iy_nominal)

    this_flux = np.sum(gal_image.array)

    # setup obj truth
    dtype = [
        ('id', int),
        ('class', np.dtype('U7')),
        ('ra', float),
        ('dec', float),
        ('x', float),
        ('y', float),
        (f'crates_flux_{band}', float),
        (f'adu_flux_{band}', float),
        (f'stamp_flux_{band}', float),
        ('COADD_OBJECT_ID', int),
        ('SOF_CM_MAG_CORRECTED_G', float),
        ('SOF_CM_MAG_CORRECTED_R', float),
        ('SOF_CM_MAG_CORRECTED_I', float),
        ('SOF_CM_MAG_CORRECTED_Z', float),
        ('SOF_CM_G_1', float),
        ('SOF_CM_G_2', float),
        ('SOF_CM_T', float),
        ('SOF_CM_TDBYTE', float),
        ('SOF_CM_FRACDEV', float),
        ('z', float),
        ('g1', float),
        ('g2', float),
        ]

    truth = np.recarray(1, dtype=dtype)

    truth['id'] = indx
    truth['class'] = 'cluster'
    truth['ra'] = gal_ra.deg
    truth['dec'] = gal_dec.deg
    truth['x'] = image_pos.x
    truth['y'] = image_pos.y
    truth['z'] = gal_z
    truth[f'crates_flux_{band}'] = obj[f'crates_adu_{band}']
    truth[f'adu_flux_{band}'] = flux_adu
    truth[f'stamp_flux_{band}'] = this_flux

    # loop through DES cols
    des_cols = [
        'COADD_OBJECT_ID',
        'SOF_CM_MAG_CORRECTED_G',
        'SOF_CM_MAG_CORRECTED_R',
        'SOF_CM_MAG_CORRECTED_I',
        'SOF_CM_MAG_CORRECTED_Z',
        'SOF_CM_G_1',
        'SOF_CM_G_2',
        'SOF_CM_T',
        'SOF_CM_TDBYTE',
        'SOF_CM_FRACDEV',
        ]
    for col in des_cols:
        truth[col] = obj[col]

    # we'll want it as a Row obj later
    truth = Table(truth)[0]

    return gal_image, truth

def sample_a_cluster(catalog, logm, z, logm_binsize=0.1, z_binsize=0.01,
                     mass_col='MASS', z_col='Z_LAMBDA', rng=None):
    '''
    Given a cluster catalog and a desired mass & redshift, sample the catalog
    for a typical cluster realization at those values, including member galaxies
    '''

    dm = logm_binsize / 2.
    dz = z_binsize / 2.
    min_m = logm - dm
    max_m = logm + dm
    min_z = z - dz
    max_z = z + dz

    # first, screen by redshift
    candidates = catalog[
        (min_z < catalog[z_col]) &\
        (catalog[z_col] < max_z)
        ]

    # second, assign masses
    candidates = assign_masses(candidates)

    # third, screen by mass
    candidates = candidates[
        (min_m < candidates['log10_mass']) &\
        (candidates['log10_mass'] < max_m)
        ]

    if len(candidates) == 0:
        raise ValueError('No cluster candidates within bounds!')

    # fourth, sample candidates
    if rng is None:
        cluster = np.random.choice(candidates, size=1)
    else:
        cluster = rng.choice(candidates, size=1)

    return Table(cluster)

def assign_masses(clusters, log_M0=14.489, F=1.356, G=-0.30, lam0=40.,
                  z0=0.35, lambda_col='LAMBDA_CHISQ', z_col='Z_LAMBDA'):
    '''
    Assign a mass estimate to each cluster given its richness
    and an assumed mass-richness relation
    Return result is predicted mass of cluster, in solar masses.
    Not sure how best to include the mass-richness parameters here,
    but the form and numbers from Tom's work:
    https://ui.adsabs.harvard.edu/abs/arXiv:1805.00039

    M = M0 * (lam/lam0)^F_lambda * [(1+z) / (1+z0)]^G_z
    with MAP parameter values given in function signature

    cluster: redmapper cat (clusters only, not members)
    '''

    richness = clusters[lambda_col]
    z = clusters[z_col]

    log10_mass = log_M0 +\
                 (F * np.log10(richness / lam0)) +\
                 (G * np.log10((1 + z) / (1 + z0)) )

    clusters['log10_mass'] = log10_mass

    return clusters

def add_des_fluxes(table, bands='griz', base='SOF_CM_MAG'):

    for band in bands:
        mag = table[f'{base}_{band.upper()}']
        flux = mag2flux(mag, 30)
        flux_col = base.replace('MAG', 'FLUX') + f'_{band}'
        table[flux_col] = flux

    return table

def flux2mag(flux, zp):
    return -2.5*np.log10(flux) + zp

def mag2flux(mag, zp):
    return np.power(10, -0.4*(mag-zp))

def get_sb_pivots():
    # Dict for SB pivot wavelengths
    piv_dict = {
        'u': 395.35082727585194,
        'b': 476.22025867791064,
        'g': 596.79880208687230,
        'r': 640.32425638312820,
        'nir': 814.02475812251110,
        'lum': 522.73829660009810
    }
    return piv_dict

def des2superbit(cat, telescope, camera, bandpass, exp_time, des_bands='griz',
                 sb_bands=['u', 'b', 'g', 'r', 'lum', 'nir']):

    for sb_band in sb_bands:
        sb_mags = des2superbit_mag(cat, sb_band)

        bandpass.transmission = phot.get_transmission(band=sb_band)

        # abmag to flux density (using closest DES band)
        mean_fnu = phot.abmag_to_mean_fnu(
            abmag=sb_mags
            )

        mean_flambda = phot.mean_flambda_from_mean_fnu(
            mean_fnu=mean_fnu,
            bandpass_transmission=bandpass.transmission,
            bandpass_wavelengths=bandpass.wavelengths
            )

        crate_electrons_pix = phot.crate_from_mean_flambda(
            mean_flambda=mean_flambda,
            illum_area=telescope.illum_area.value,
            bandpass_transmission=bandpass.transmission,
            bandpass_wavelengths=bandpass.wavelengths
            )

        # ADU counts per second
        crate_adu_pix = crate_electrons_pix / camera.gain.value

        # ADU counts per exposure time (what is used in sims)
        flux_adu = crate_adu_pix * exp_time * 1

        cat[f'crates_adu_{sb_band}'] = crate_adu_pix
        cat[f'flux_adu_{sb_band}_{exp_time}'] = flux_adu

    return cat

# def get_closest_des_mag(sb_band):
def des2superbit_mag(cat, sb_band):
    '''
    Approximate the superbit mag given a superbit filter passband

    as the bands are not defined to be the same, we interpolate between
    the two closest DES bands

    NOTE: a bit hacky, but close enough for our use case
    '''

    sb_pivot = get_sb_pivots()[sb_band] # nm

    # NOTE: approximate values from NOIRLab
    # https://noirlab.edu/science/programs/ctio/filters/Dark-Energy-Camera

    # NOTE: computed quantities taken from:
    # http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=CTIO/DECam.u&&mode=browse&gname=CTIO&gname2=DECam
    DES_FILTERS = {
        'u': {
            'blue': 312,
            'red': 400,
            'pivot': 381.433,
            'mean_transmission': 0.15,
        },
        'g': {
            'blue': 398,
            'red': 548,
            'pivot': 480.849,
            'mean_transmission': 0.5,
        },
        'r': {
            'blue': 568,
            'red': 716,
            'pivot': 641.765,
            'mean_transmission': 0.75,
        },
        'i': {
            'blue': 710,
            'red': 857,
            'pivot': 781.458,
            'mean_transmission': 0.9,
        },
        'z': {
            'blue': 850,
            'red': 1002,
            'pivot': 916.886,
            'mean_transmission': 0.58,
        },
    }

    des_mag_col_base = 'SOF_CM_MAG_CORRECTED_'

    # we only use griz as those are the filters we have cluster member
    # photometry for
    use_bands = 'griz'
    des_pivots = [DES_FILTERS[b]['pivot'] for b in use_bands]

    mag_est = np.zeros(len(cat))
    for i, gal in enumerate(cat):
        des_fluxes = [mag2flux(gal[f'{des_mag_col_base}{b.upper()}'], 30) for b in use_bands]

        # sp.interp1d(des_pivots, des_fluxes)
        flux_est = np.interp(sb_pivot, des_pivots, des_fluxes)
        mag_est[i] = flux2mag(flux_est, 30)

    return mag_est
