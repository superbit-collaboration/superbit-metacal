#Standard imports
import galsim
import numpy as np
import math
import logging

# Modularized Galsim code related imports
from tools import truth, lensing_tools

__all__ = ["make_bkg_galaxy"]

def make_bkg_galaxy(ud, wcs, jitter_psf, 
                    affine, fits_cat_data, 
                    cosmos_cat, nfw_halo, sb_optical_psf, 
                    bandpass, sbparams, logger):
    """ Function to make a background galaxy stamp

    Args:
        ud ([type]): [description]
        wcs ([type]): [description]
        affine ([type]): [description]
        fits_cat_data ([type]): [description]
        cosmos_cat ([type]): [description]
        nfw_halo ([type]): [description]
        sb_optical_psf ([type]): [description]
        bandpass ([type]): [description]
        sbparams ([type]): [description]
    
    Return:
        stamp: Stamp image of the bkg galaxy
        bkg_gal_truth: True attributes of the bkg galaxy on the stamp
    """
    # For the position of the background galaxy, choose a random 
    # RA and DEC around the sky center. Note that for this to come out
    # close to a square shape, we need to account for the cos(dec) part of
    # the metric: ds^2 = dr^2 + r^2 * d(dec)^2 + r^2 * cos^2(dec) * d(ra)^2

    # Calculate the RA and DEC position in arcsec
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

    # Calculate the image position from the WCS.
    image_pos = wcs.toImage(world_pos)

    # Calculate the tangent plane, which we refere to the "world coordinates"
    # here. This is still an x/y coordinate
    uv_pos = affine.toWorld(image_pos)

    # Create a chromatic background galaxy
    bkg_gal = cosmos_cat.makeGalaxy(gal_type='parametric',
                                    rng=ud,
                                    chromatic=True)
    
    # Extract galaxy redshift from the COSMOS profile fit catalog
    bkg_gal_z = fits_cat_data['zphot'][bkg_gal.index]

    # NOTE THIS IS SOMETHING I NEED TO FIGURE OUT. THERE CANNOT BE
    # BKG GALAXIES IN THE CATALOG ABOVE AT LOWER Z THAN THE CLUSTER ITSELF 
    if bkg_gal_z <= sbparams.nfw_z_cluster:
        bkg_gal_z = 0.5

    # Apply a random rotation to the bkg galaxy created. Why? I suppose
    # this is just to randomize their orientations. 
    theta = ud()*2.0*np.pi*galsim.radians
    bkg_gal = bkg_gal.rotate(theta)

    # The next line is to rescale the flux of the created galaxy
    # from the flux observed by a 1 second exposure with HST to an n second
    # exposure with SuperBIT. HOWEVER, THIS SCALING FACTOR IS STILL NOT FULLY 
    # CERTAIN, SO PROCEED WITH CAUTION. 
    bkg_gal *= sbparams.flux_scaling

    # Obtain the reduced shear & magnification as expected by the simulated
    # NFW halo at the given position of the background galaxy.
    reduced_shear, mu = lensing_tools.nfw_lensing(nfw_halo=nfw_halo, 
                                                  pos=uv_pos,
                                                  nfw_z_source=bkg_gal_z)
    g1, g2 = reduced_shear.g1, reduced_shear.g2

    # Now that we have the expected reduced shear and magnification
    # for the bkg galaxy, lens the bkg galaxy with the expected shear & 
    # magnification for consistency. 

    try:
        bkg_gal = bkg_gal.lens(g1, g2, mu)
    except galsim.errors.GalSimError:
        print("Could not lens background galaxy, setting default values...")
        g1, g2 = 0., 0.
        mu = 1.

    # Convolve the optical PSF, jitter PSF, and the galaxy flux
    jitter_psf = jitter_psf.getPSF(image_pos)
    gsparams = galsim.GSParams(maximum_fft_size=16384)

    convolution = galsim.Convolve([jitter_psf, 
                                   sb_optical_psf,
                                   bkg_gal], 
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

    # Now we want to create a "stamp" sized image of the background galaxy
    stamp_size_x = 64 # pixels
    stamp_size_y = 64 # pixels

    stamp = galsim.Image(stamp_size_x, stamp_size_y, 
                         wcs=wcs.local(image_pos))
    stamp = convolution.drawImage(bandpass=bandpass,
                                  image=stamp,
                                  offset=offset,
                                  method='no_pixel')
    stamp.setCenter(ix_nominal, iy_nominal)

    # Assign the "true" attributes of the stamp for reference
    bkg_gal_truth = truth.true_attributes()
    bkg_gal_truth.ra = ra.deg
    bkg_gal_truth.dec = dec.deg
    bkg_gal_truth.x = ix_nominal
    bkg_gal_truth.y = iy_nominal
    bkg_gal_truth.g1 = g1
    bkg_gal_truth.g2 = g2
    bkg_gal_truth.mu = mu
    bkg_gal_truth.z = bkg_gal_z
    bkg_gal_truth.flux = stamp.added_flux

    try:
        bkg_gal_truth.fwhm = (convolution.evaluateAtWavelength(
                                         sbparams.lam).calculateFWHM())
    except galsim.errors.GalSimError:
        logger.debug("FWHM calculation failed. Setting FWHM = -9999.0")
        bkg_gal_truth.fwhm = -9999.0

    try:
        bkg_gal_truth.mom_sigma = stamp.FindAdaptiveMom().moments_sigma
    except galsim.errors.GalSimError:
        logger.debug("Moments sigma calculation failed. Setting to -9999.0")
        bkg_gal_truth.mom_sigma = -9999.0
    
    return stamp, bkg_gal_truth




                                        
                                    



    

    
    


