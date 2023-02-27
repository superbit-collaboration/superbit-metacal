import numpy as np
import galsim
import math
from astropy.table import Table, Row

import ipdb

def make_a_galaxy(indx, obj, band, wcs, psf, nfw_halo, camera, exp_time,
                  pix_scale, cosmos_plate_scale, ra_bounds, dec_bounds,
                  gs_params, logprint):
    '''
    indx: source index
    gal: np.recarray row
    camera: container for needed camera quantities
    '''

    if indx % 100 == 0:
        logprint(f'Starting {indx}')

    # determine if we should skip
    ra_min, ra_max = ra_bounds
    dec_min, dec_max = dec_bounds

    gal_ra = obj['ra']
    gal_dec = obj['dec']

    if (ra_min.value > gal_ra) or (ra_max.value < gal_ra) or\
       (dec_min.value > gal_dec) or (dec_max.value < gal_dec):
        logprint(f'gal {indx} out of bounds. Skipping.')
        return (None, None)

    # Get its Sersic parameters
    gal_n_sersic_cosmos10 = obj['c10_sersic_fit_n']

    if gal_n_sersic_cosmos10 < 0.3:
        gal_n_sersic_cosmos10 = 0.3

    crate_flux = obj[f'crates_{band}'] # galaxy count rate e/s

    flux_adu = int(
        crate_flux * exp_time * 1 / camera.gain.value
        )

    hlr_arcsec = (obj['c10_sersic_fit_hlr']
                * np.sqrt(obj['c10_sersic_fit_q'])
                * cosmos_plate_scale)

    if hlr_arcsec <= 0:
        return (None, None)

    hlr_pixels = hlr_arcsec / pix_scale

    gal = galsim.Sersic(
        n=gal_n_sersic_cosmos10,
        half_light_radius=hlr_arcsec,
        flux=flux_adu
        )

    gal = gal.shear(
        q=obj['c10_sersic_fit_q'],
        beta=obj['c10_sersic_fit_phi']*galsim.radians
        )

    gal_ra *= galsim.degrees
    gal_dec *= galsim.degrees

    world_pos = galsim.CelestialCoord(gal_ra, gal_dec)
    image_pos = wcs.toImage(world_pos)

    # galaxy 2d position on sci img in arcsec
    gal_pos_cent_xasec = image_pos.x * pix_scale
    gal_pos_cent_yasec = image_pos.y * pix_scale

    gal_z = obj['ZPDF']

    # get the expected shear from the halo at galaxy 2d position
    try:
        g1, g2 = nfw_halo.getShear(
            pos=galsim.PositionD(
                x=gal_pos_cent_xasec,
                y=gal_pos_cent_yasec
                ),
            z_s=gal_z,
            units=galsim.arcsec,
            reduced=True
            )

        nfw_mu = nfw_halo.getMagnification(
            pos=galsim.PositionD(
                x=gal_pos_cent_xasec,
                y=gal_pos_cent_yasec
                ),
            z_s=gal_z,
            units=galsim.arcsec
            )

    except galsim.errors.GalSimRangeError:
        g1, g2 = 0,0
        nfw_mu = 1.0

    abs_val = np.sqrt(g1**2 + g2**2)

    if abs_val >= 1:  # strong lensing
        g1 = 0
        g2 = 0

    # strong lensing
    if nfw_mu < 0:
        logprint('Warning: mu < 0 means strong lensing! Using mu=25.')
        nfw_mu = 25
    elif nfw_mu > 25:
        logprint('Warning: mu > 25 means strong lensing! Using mu=25.')
        nfw_mu = 25

    # lens the Sersic object galaxy
    gal = gal.lens(g1=g1, g2=g2, mu=nfw_mu)

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
    # stamp_size = round(hlr_pixels + 500)

    gal_image = convolution.drawImage(
        # nx=stamp_size,
        # ny=stamp_size,
        wcs=wcs.local(image_pos),
        offset=offset,
        method='auto',
        # dtype=np.uint16
        )

    gal_image.setCenter(ix_nominal, iy_nominal)

    this_flux = np.sum(gal_image.array)

    # setup obj truth
    dtype = [
        ('id', int),
        ('class', np.dtype('U4')),
        ('ra', float),
        ('dec', float),
        ('x', float),
        ('y', float),
        ('sersic_n', float),
        (f'crates_flux_{band}', float),
        (f'adu_flux_{band}', float),
        (f'stamp_flux_{band}', float),
        ('hlr', float),
        ('z', float),
        ('g1', float),
        ('g2', float),
        ('mu', float),
        ]

    truth = np.recarray(1, dtype=dtype)

    truth['id'] = indx
    truth['class'] = 'gal'
    truth['ra'] = gal_ra.deg
    truth['dec'] = gal_dec.deg
    truth['x'] = image_pos.x
    truth['y'] = image_pos.y
    truth['sersic_n'] = gal_n_sersic_cosmos10
    truth['hlr'] = hlr_arcsec
    truth['z'] = gal_z
    truth['g1'] = g1
    truth['g2'] = g2
    truth['mu'] = nfw_mu
    truth[f'crates_flux_{band}'] = crate_flux
    truth[f'adu_flux_{band}'] = flux_adu
    truth[f'stamp_flux_{band}'] = this_flux

    # we'll want it as a Row obj later
    truth = Table(truth)[0]

    return gal_image, truth
