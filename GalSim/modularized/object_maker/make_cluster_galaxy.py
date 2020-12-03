#Standard imports
import galsim
import numpy as np
import math

# Modularized Galsim code related imports
from tools import truth

__all__ = ["make_cluster_galaxy"]

def make_cluster_galaxy(ud, wcs, jitter_psf, affine, centerpix,
                        cluster_cat, sb_optical_psf, bandpass,
                        sbparams, radius, logger):
    """Function to make a cluster galaxy stamp

    Args:
        ud ([type]): [description]
        wcs ([type]): [description]
        jitter_psf ([type]): [description]
        affine ([type]): [description]
        centerpix ([type]): [description]
        cluster_cat ([type]): [description]
        sb_optical_psf ([type]): [description]
        bandpass ([type]): [description]
        sbparams ([type]): [description]
        radius ([type]): [description]
    """

    # Choose a random position within 200 pixels of the sky center
    max_rsq = radius**2.

    while True:  # (This is essentially a do..while loop.)
        x = (2.*ud()-1) * radius 
        y = (2.*ud()-1) * radius 
        rsq = x**2 + y**2
    
        if rsq <= max_rsq: break
    
    # We need the image position as well, so use the wcs to get that.
    # Add some Gaussian jitter, so the cluster does not look too box-like
    image_pos = galsim.PositionD(x + centerpix.x + (ud()-0.5) * 10,
                                 y + centerpix.y + (ud()-0.5) * 10)
    world_pos = wcs.toWorld(image_pos)
    ra = world_pos.ra
    dec = world_pos.dec

    # We also need this in the tangent plane, which we call 
    # "world coordinates" here, this is still an x/y corrdinate 
    uv_pos = affine.toWorld(image_pos)

    # FIXME: This appears to be missing and should be fixed????
    # JAVIER POSTED THIS. NOT SURE OF THE STATUS??

    # Assign zero shear to cluster galaxies?
    g1 = 0.0; g2 = 0.0
    mu = 1.0

    # Create chromatric cluster galaxy
    cluster_gal = cluster_cat.makeGalaxy(gal_type='parametric',
                                         rng=ud,
                                         chromatic=True)
    logger.debug('Created cluster galaxy')

    # Apply a random rotation to the cluster galaxy
    theta = ud()*2.*np.pi*galsim.radians
    cluster_gal = cluster_gal.rotate(theta)

    # Scale flux between HST and SuperBIT. CAUTION: not sure if this is 
    # correct yet
    cluster_gal *= sbparams.flux_scaling
    cluster_gal.magnify(10)

    # Convolve the optical PSF, jitter PSF, and the galaxy flux
    jitter_psf = jitter_psf.getPSF(image_pos)
    gsparams = galsim.GSParams(maximum_fft_size=16384)

    convolution = galsim.Convolve([jitter_psf, 
                                   sb_optical_psf,
                                   cluster_gal], 
                                   gsparams=gsparams)
    # Account for the fractional part of the position
    # cf. demo9.py for an explanation of this nominal position stuff.
    x_nominal = image_pos.x + 0.5
    y_nominal = image_pos.y + 0.5
    ix_nominal = int(math.floor(x_nominal+0.5))
    iy_nominal = int(math.floor(y_nominal+0.5))
    dx = x_nominal - ix_nominal
    dy = y_nominal - iy_nominal
    offset = galsim.PositionD(dx,dy)

    # Draw cluster galaxy stamp
    stamp_size_x = 128
    stamp_size_y = 128

    stamp = galsim.Image(stamp_size_x, 
                         stamp_size_y, 
                         wcs=wcs.local(image_pos))
    stamp = convolution.drawImage(bandpass=bandpass,
                                  image=stamp, 
                                  offset=offset,
                                  method='no_pixel')
    stamp.setCenter(ix_nominal,iy_nominal)

    # Assign the "true" attributes of the stamp for reference
    cluster_gal_truth = truth.true_attributes()
    cluster_gal_truth.ra = ra.deg
    cluster_gal_truth.dec = dec.deg
    cluster_gal_truth.x = ix_nominal
    cluster_gal_truth.y = iy_nominal
    cluster_gal_truth.g1 = g1
    cluster_gal_truth.g2 = g2
    cluster_gal_truth.mu = mu
    cluster_gal_truth.z = sbparams.nfw_z_cluster
    cluster_gal_truth.flux = stamp.added_flux

    try:
        cluster_gal_truth.fwhm = (convolution.evaluateAtWavelength(
                                         sbparams.lam).calculateFWHM())
    except galsim.errors.GalSimError:
        logger.debug("FWHM calculation failed. Setting FWHM = -9999.0")
        cluster_gal_truth.fwhm = -9999.0

    try:
        cluster_gal_truth.mom_sigma = stamp.FindAdaptiveMom().moments_sigma
    except galsim.errors.GalSimError:
        logger.debug("Moments sigma calculation failed. Setting to -9999.0")
        cluster_gal_truth.mom_sigma = -9999.0
    
    return stamp, cluster_gal_truth



