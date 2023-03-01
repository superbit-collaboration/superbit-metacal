import numpy as np
import galsim
import math
from astropy.table import Table, Row

import ipdb

def make_a_star(indx, obj, band, wcs, psf, camera, exp_time, pix_scale,
                ra_bounds, dec_bounds, gsparams, stamp_size, logprint):
    '''
    Make a single GAIA star given inputs. Setup for multiprocessing

    indx: int
        The batch index
    obj: np.recarray, astropy.Table row
        The stellar object to simulate
    band: str
        The band to simulate
    wcs: galsim.WCS
        The image WCS
    camera: instrument.Camera
        An instance of the Camera class
    exp_time: int, float
        The exposure time

    returns:
    '''

    this_flux_adu = obj[f'flux_adu_{band}']

    # determine if we should skip
    ra_min, ra_max = ra_bounds
    dec_min, dec_max = dec_bounds

    star_ra = obj['RA_ICRS'] * galsim.degrees
    star_dec = obj['DE_ICRS'] * galsim.degrees

    if (ra_min.value > star_ra.deg) or (ra_max.value < star_ra.deg) or\
       (dec_min.value > star_dec.deg) or (dec_max.value < star_dec.deg):
        logprint(f'star {indx} out of bounds. Skipping')

        return (None, None)

    # Assign real position to the star on the sky
    world_pos = galsim.CelestialCoord(
        star_ra, star_dec
        )
    image_pos = wcs.toImage(world_pos)

    star = galsim.DeltaFunction(flux=this_flux_adu)

    # Position fractional stuff
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5

    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))

    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal

    offset = galsim.PositionD(dx, dy)

    # TODO: get this to work according to Mike's suggestion!
    # Update the PSF to account for roll angle
    # ipdb.set_trace()
    # psf_roll = wcs.toImage(psf, image_pos=image_pos)

    # convolve star with the psf
    convolution = galsim.Convolve(
        [psf, star], gsparams=gsparams
        )

    star_image = convolution.drawImage(
        nx=stamp_size,
        ny=stamp_size,
        wcs=wcs.local(image_pos),
        offset=offset,
        method='auto',
        # dtype=np.uint16
        )

    star_image.setCenter(ix_nominal, iy_nominal)

    # setup obj truth
    dtype = [
        ('id', int),
        ('class', np.dtype('U4')),
        ('ra', float),
        ('dec', float),
        ('x', float),
        ('y', float),
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
    truth['class'] = 'star'
    truth['ra'] = star_ra.deg
    truth['dec'] = star_dec.deg
    truth['x'] = image_pos.x
    truth['y'] = image_pos.y
    truth[f'stamp_flux_{band}'] = this_flux_adu

    # we'll want it as a Row obj later
    truth = Table(truth)[0]

    return star_image, truth
