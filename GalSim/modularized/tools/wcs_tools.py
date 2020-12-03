import galsim
import numpy as np

__all__ = ["get_affine_wcs"]

def get_affine_wcs(theta, sbparams, image_center):
    """Returns affine for a given theta, pixel scale, and image center

    Args:
        theta (float): rotation angle for the non-trivial wcs system (degrees)
        sbparams (class): SuperBIT parameters as defined in the config file 
        image_center (galsim.position.PositionD class property)
    
    Returns: (tuple)
        index 0: affine (galsim.wcs.AffineTransform class property)
        index 1: wcs_img
    """
    
    theta = 0.0 * galsim.degrees
    dudx = np.cos(theta) * sbparams.pixel_scale
    dudy = -np.sin(theta) * sbparams.pixel_scale
    dvdx = np.sin(theta) * sbparams.pixel_scale
    dvdy = np.cos(theta) * sbparams.pixel_scale
    
    affine = galsim.AffineTransform(dudx, dudy, dvdx, dvdy, 
                                    origin=image_center)

    sky_center =  galsim.CelestialCoord(ra=sbparams.center_ra, 
                                        dec=sbparams.center_dec)      

    wcs_img =  galsim.TanWCS(affine, sky_center, units=galsim.arcsec)
    return affine, wcs_img


