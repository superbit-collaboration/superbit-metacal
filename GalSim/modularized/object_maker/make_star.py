# Standard imports
import galsim
import numpy as np
import math

# Modularized Galsim code related imports
from tools import truth

__all__ = ["make_star"]

def make_star(ud, wcs, jitter_psf, 
              affine, sb_optical_psf, 
              sbparams):
    """ Makes a star stamp.

    Args:
        ud ([type]): [description]
        wcs ([type]): [description]
        jitter_psf ([type]): [description]
        affine ([type]): [description]
        sb_optical_psf ([type]): [description]
        sbparams ([type]): [description]
        logger ([type]): [description]
    """

    # Choose random RA and DEC around the sky center
    dec = (sbparams.center_dec
            + (ud()-0.5) 
            * sbparams.image_ysize_arcsec 
            * galsim.arcsec)
    ra = (sbparams.center_ra 
           + (ud()-0.5) 
           * sbparams.image_xsize_arcsec 
           / np.cos(dec) 
           * galsim.arcsec)
    world_pos = galsim.CelestialCoord(ra, dec)
    image_pos = wcs.toImage(world_pos)

    # Draw star flux at random; based on distribution of 
    # star fluxes in real images  
    flux_dist = galsim.DistDeviate(ud, 
                                   function = lambda x:x**-1.5, 
                                   x_min = 799.2114, 
                                   x_max = 890493.9)
    star_flux = flux_dist()

    # Generate PSF at location of star, convolve with optical 
    # model to make a star
    deltastar = galsim.DeltaFunction(flux=star_flux)  
    jitter_psf = jitter_psf.getPSF(image_pos)
    convolution = galsim.Convolve([sb_optical_psf, jitter_psf,deltastar])
        
    # Account for the fractional part of the position
    # cf. demo9.py for an explanation of this nominal position stuff.
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)

    star_stamp = convolution.drawImage(wcs=wcs.local(image_pos), 
                                       offset=offset, 
                                       method='no_pixel')
    star_stamp.setCenter(ix_nominal,iy_nominal)

    star_truth = truth.true_attributes()
    star_truth.ra = ra.deg
    star_truth.dec = dec.deg
    star_truth.x = ix_nominal
    star_truth.y = iy_nominal

    return star_stamp, star_truth
